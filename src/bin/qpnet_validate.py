#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Wu Yi-Chiao (Nagoya University)
# based on a WaveNet script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/PytorchWaveNetVocoder)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
import argparse
import logging
import os
import sys
import time
import six
import torch
import numpy as np
import yaml
import copy

from dateutil.relativedelta import relativedelta
from distutils.util import strtobool
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import transforms
from scipy.io import wavfile

from utils import background
from utils import extend_time
from utils import find_files
from utils import read_hdf5
from utils import read_txt
from utils import check_filenames

from qpnet import encode_mu_law
from qpnet import initialize
from qpnet import QPNet

    
def _get_arguments():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--waveforms", required=True,
                        type=str, help="directory or list of validation wav files")
    parser.add_argument("--feats", required=True,
                        type=str, help="directory or list of validation aux feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="hdf5 file including statistics")
    parser.add_argument("--resultdir", required=True,
                        type=str, help="directory to save the validatation results")
    parser.add_argument("--config", required=True,
                        type=str, help="path of the model config")
    parser.add_argument("--checkpoint", required=True,
                        type=str, help="model file")
    # network training setting
    parser.add_argument("--batch_length", default=20000,
                        type=int, help="batch length")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="batch size")
    parser.add_argument("--max_length", default=30000,
                        type=int, help="maximum length of batach and receptive field")
    parser.add_argument("--f0_threshold", default=0,
                        type=int, help="threshold of lowest f0")
    # other setting
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")

    return parser.parse_args()

def _validate_length(x, y, upsampling_factor=None):
    """VALIDATE LENGTH
    Args:
        x (ndarray): numpy.ndarray with x.shape[0] = len_x
        y (ndarray): numpy.ndarray with y.shape[0] = len_y
        upsampling_factor (int): upsampling factor
    Returns:
        (ndarray): length adjusted x with same length y
        (ndarray): length adjusted y with same length x
    """
    if upsampling_factor is None:
        if x.shape[0] < y.shape[0]:
            y = y[:x.shape[0]]
        if x.shape[0] > y.shape[0]:
            x = x[:y.shape[0]]
        assert len(x) == len(y)
    else:
        if x.shape[0] > y.shape[0] * upsampling_factor:
            x = x[:y.shape[0] * upsampling_factor]
        if x.shape[0] < y.shape[0] * upsampling_factor:
            mod_y = y.shape[0] * upsampling_factor - x.shape[0]
            mod_y_frame = mod_y // upsampling_factor + 1
            y = y[:-mod_y_frame]
            x = x[:y.shape[0] * upsampling_factor]
        assert len(x) == len(y) * upsampling_factor

    return x, y

def _dilated_factor(batch_f0, fs, dense_factor):
    """PITCH-DEPENDENT DILATED FACTOR
    Args:
        batch_f0 (numpy array): the f0 sequence (T)
        fs (int): sampling rate
        dense_factor (int): the number of taps in one cycle
    Return:
        dilated_factors(np array): 
            float array of the pitch-dependent dilated factors (T)
    """
    f0s = copy.deepcopy(batch_f0)
    f0s[f0s == 0]   = fs/dense_factor
    dilated_factors = np.ones(f0s.shape)*fs
    dilated_factors /= f0s
    dilated_factors /= dense_factor
    assert np.all(dilated_factors > 0)
    return dilated_factors

def _batch_f0(h, f0_threshold = 0):
    """LOAD F0 SEQUENCE
    Args:
        h (numpy array): the auxiliary acoustic features (T x D)
        f0_threshold (float): the lower bound of pitch
    Return:
        cont_f0_lpf(numpy array): 
            float array of the continuous pitch sequence (T)
    """
    #uv = h[:, 0].copy(order='C')  # voive/unvoice feature
    cont_f0_lpf = h[:, 1].copy(order='C')  # continuous f0
    #mcep = h[:, 2:feat_param['mcep_dim_end']].copy(order='C')  # mcc
    #codeap = h[:, feat_param['codeap_index']:].copy(order='C')  # coded ap
    cont_f0_lpf[cont_f0_lpf < f0_threshold] = f0_threshold
    return cont_f0_lpf

def _receptive_field(receptiveCausal_field,
                     receptiveF_field,
                     receptiveA_field,
                     dilated_factors):
    """GET RECEPTIVE FILED
    Args:
        receptiveCausal_field (int): receptive field of causal layer
        receptiveF_field (int): receptive field of fixed network
        receptiveA_field (int): receptive field of adaptive network
        dilated_factors(np array): 
            float array of the pitch-dependent dilated factors (T)
    Return:
        receptive_field (int): receptive field of whole network
    """
    max_dilation_factor = np.nanmax(dilated_factors)
    receptiveA_field *= int(np.ceil(max_dilation_factor))
    #print("receptiveA_field: %f, max_dilation: %f"%(receptiveA_field, max_dilation_factor))
    return int(receptiveF_field + receptiveA_field + receptiveCausal_field)

@background(max_prefetch=2)
def validate_generator(wav_list,
                       feat_list,
                       model_receptiveCausal,
                       model_receptiveF,
                       model_receptiveA,
                       wav_transform=None,
                       feat_transform=None,
                       feature_type="world",
                       dense_factor=8,                    
                       batch_length=20000,
                       batch_size=1,
                       max_length=23070,
                       f0_threshold=0,
                       upsampling_factor=80):
    """VALIDATION BATCH GENERATOR
    Args:
        wav_list (str): list of wav files
        feat_list (str): list of feat files
        model_receptiveCausal (int): receptive field length of causal layer
        model_receptiveF (int): receptive field length of fixed module
        model_receptiveA (int): receptive field length of adaptive module
        wav_transform (func): preprocessing function for waveform
        feat_transform (func): preprocessing function for auxiliary feature
        feature_type (str): auxiliary feature type
        dense_factor (int): dilation taps in one cycle
        batch_length (int): batch length 
        batch_size (int): batch size 
        max_length (int): maximum of (batch length + receptive filed length)
        f0_threshold (int): threshold of lowest f0 (f0=f0_threshold, when f0 < f0_threshold)
        upsampling_factor (int): upsampling factor
    Return:
        (object): generator instance
    """
    batch_x, batch_h, batch_t = [], [], []
    batch_d, batch_b = [], []
    batch_count = batch_size
    # process over all of files
    for wavf, featf in zip(wav_list, feat_list):
        assert check_filenames([wavf, featf])
        # load wavefrom and aux feature
        fs, x = wavfile.read(wavf)
        x = np.array(x, dtype=np.float32)/32768
        h = read_hdf5(featf, "/%s" % feature_type)
        x, h = _validate_length(x, h, upsampling_factor)
        d = _dilated_factor(_batch_f0(h, f0_threshold), fs, dense_factor)
        d = np.squeeze(extend_time(np.expand_dims(d, -1), upsampling_factor), -1)
        # ------------------------------------
        # use mini batch with upsampling layer
        # ------------------------------------
        # make buffer array
        if "x_buffer" not in locals():
            x_buffer = np.empty((0), dtype=np.float32)
            h_buffer = np.empty((0, h.shape[1]), dtype=np.float32)
            d_buffer = np.empty((0), dtype=np.float32)
        x_buffer = np.concatenate([x_buffer, x], axis=0)
        h_buffer = np.concatenate([h_buffer, h], axis=0)
        d_buffer = np.concatenate([d_buffer, d], axis=0)
        # get current receptive field length
        receptive_field = _receptive_field(
            receptiveCausal_field=model_receptiveCausal,
            receptiveF_field=model_receptiveF,
            receptiveA_field=model_receptiveA,
            dilated_factors=d_buffer)
        # adjust the batch_length to avoid GPU out of memory issue
        batch_mod1 = max((receptive_field + batch_length - max_length), 0)
        batch_length_current = batch_length - batch_mod1
        # adjust the batch_length to meet upsampling ratio
        batch_mod2 = (receptive_field + batch_length_current) % upsampling_factor
        batch_length_current -= batch_mod2
        if batch_mod1 + batch_mod2:
            logging.debug("batch length is decreased (%d -> %d)" % (
                batch_length, batch_length_current))
        # set batch length
        h_bs = (receptive_field + batch_length_current) // upsampling_factor
        x_bs = h_bs * upsampling_factor + 1
        while len(h_buffer) > (batch_count * h_bs) and len(x_buffer) > (batch_count * x_bs):
            # get pieces
            h_ = h_buffer[:h_bs, :]
            x_ = x_buffer[:x_bs]
            d_ = d_buffer[:x_bs]
            # perform pre-processing
            if wav_transform is not None:
                x_ = wav_transform(x_)
            if feat_transform is not None:
                h_ = feat_transform(h_)
            # convert to torch variable
            x_ = torch.from_numpy(x_).long()
            h_ = torch.from_numpy(h_).float()
            d_ = torch.from_numpy(d_).float()
            # remove the last and first sample for training
            batch_h += [h_.transpose(0, 1)]  # (D x T)
            batch_x += [x_[:-1]]  # (T)
            batch_t += [x_[1:]]  # (T)
            batch_d += [d_[:-1]]  # (B x T)
            batch_b += [batch_length_current]
            batch_count -= 1
            # set shift size
            h_ss = batch_length_current // upsampling_factor
            x_ss = h_ss * upsampling_factor
            # update buffer
            h_buffer = h_buffer[h_ss:, :]
            x_buffer = x_buffer[x_ss:]
            d_buffer = d_buffer[x_ss:]
            # return mini batch
            if len(batch_x) == batch_size:
                batch_x = torch.stack(batch_x)
                batch_h = torch.stack(batch_h)
                batch_t = torch.stack(batch_t)
                batch_d = torch.stack(batch_d)
                batch_b = torch.tensor(batch_b)
                # send to cuda
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_h = batch_h.cuda()
                    batch_t = batch_t.cuda()
                    batch_d = batch_d.cuda()
                    batch_b = batch_b.cuda()
                yield batch_x, batch_h, batch_t, batch_d, batch_b
                batch_x, batch_h, batch_t = [], [], []
                batch_d, batch_b = [], []
                batch_count = batch_size

def main():
    # parser arguments
    args = _get_arguments()
    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
        logging.warn("logging is disabled.")

    # show argmument
    for key, value in vars(args).items():
        logging.info("%s = %s" % (key, str(value)))

    # check output directory
    if not os.path.exists(args.resultdir):
        os.makedirs(args.resultdir)

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # load config
    config = torch.load(args.config)
    ## if the program is very slow in some pytorch & cuda version
    ## you can try the following 2 settings
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    # define network
    upsampling_factor = config.upsampling_factor
    model = QPNet(
        n_quantize=config.n_quantize,
        n_aux=config.n_aux,
        n_resch=config.n_resch,
        n_skipch=config.n_skipch,
        dilationF_depth=config.dilationF_depth,
        dilationF_repeat=config.dilationF_repeat,
        dilationA_depth=config.dilationA_depth,
        dilationA_repeat=config.dilationA_repeat,
        kernel_size=config.kernel_size,
        upsampling_factor=upsampling_factor)
    logging.debug(model)  
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # setups for multi GPUs
    if args.n_gpus > 1:
        device_ids = range(args.n_gpus)
        model = torch.nn.DataParallel(model, device_ids)
        model.receptiveF_field = model.module.receptiveF_field
        model.receptiveA_field = model.module.receptiveA_field
        model.receptiveCausal_field = model.module.receptiveCausal_field
        if args.n_gpus > args.batch_size:
            logging.warn("batch size is less than number of gpus.")

    # define transforms
    scaler = StandardScaler()
    scaler.mean_ = read_hdf5(args.stats, "/%s/mean" % config.feature_type)
    scaler.scale_ = read_hdf5(args.stats, "/%s/scale" % config.feature_type)
    wav_transform = transforms.Compose([
        lambda x: encode_mu_law(x, config.n_quantize)])
    feat_transform = transforms.Compose([
        lambda x: scaler.transform(x)])

    # define generator
    feat_ext = ".%s" % config.feature_format # feature file extension
    if os.path.isdir(args.waveforms):
        filenames = sorted(find_files(args.waveforms, "*.wav", use_dir_name=False))
        wav_list = [args.waveforms + "/" + filename for filename in filenames]
        feat_list = [args.feats + "/" + filename.replace(".wav", feat_ext) for filename in filenames]
    elif os.path.isfile(args.waveforms):
        wav_list = read_txt(args.waveforms)
        feat_list = read_txt(args.feats)
    else:
        logging.error("--waveforms should be directory or list.")
        sys.exit(1)
    assert len(wav_list) == len(feat_list)
    logging.info("number of validation data = %d." % len(wav_list))
    logging.info("max length: %s" % str(args.max_length))
    logging.info("f0 threshold: %s" % str(args.f0_threshold))
    #logging.info("continuous f0: %s" % str(args.contf0_flag))
    assert args.batch_length > 0
    assert args.batch_size > 0    
    generator = validate_generator(
        wav_list, feat_list,
        model_receptiveCausal=model.receptiveCausal_field,
        model_receptiveF=model.receptiveF_field,
        model_receptiveA=model.receptiveA_field,
        wav_transform=wav_transform,
        feat_transform=feat_transform,
        feature_type=config.feature_type,
        dense_factor=config.dense_factor,
        batch_length=args.batch_length,
        batch_size=args.batch_size,
        max_length=args.max_length,
        f0_threshold=args.f0_threshold,
        upsampling_factor=upsampling_factor)

    # charge minibatch in queue
    while not generator.queue.full():
        time.sleep(0.1)
    
    # load model
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    logging.info("load %s." % args.checkpoint)
    if args.n_gpus > 1:
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    # check gpu and then send to gpu
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        #sys.exit(1)
    
    # loss recording
    flossyml = args.resultdir + "/validation_result.yml"
    if os.path.exists(flossyml):
        with open(flossyml, "r", encoding='utf-8') as yf:
            results_dict = yaml.safe_load(yf)
    else:
        results_dict = {}

    # train
    loss = 0
    model_name = str(os.path.basename(args.checkpoint))
    logging.info('Modle: %s' % model_name)
    for i, (batch_x, batch_h, batch_t, batch_d, batch_b) in enumerate(generator):
        batch_output = model(batch_x, batch_h, batch_d, batch_b)

        assert torch.all(batch_b==batch_b[0])
        batch_length_current = int(batch_b[0]) # shift for each layer

        assert(torch.max(batch_t)<config.n_quantize)
        batch_loss = criterion(
            batch_output[:, -batch_length_current:].contiguous().view(-1, config.n_quantize),
            batch_t[:, -batch_length_current:].contiguous().view(-1))

        loss += batch_loss.item()
        logging.info("(batch:%d) batch loss = %.6f" % (i + 1, batch_loss.item()))

    results_dict.update({model_name: float(loss / (i+1))})      
    # save the loss record
    with open(flossyml, "w", encoding='utf-8') as yf:
        yaml.safe_dump(results_dict, yf)


if __name__ == "__main__":
    main()
