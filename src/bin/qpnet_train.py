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

def count_parameters(model):
    """NUMBER OF MODEL PARAMETER
    Args:
        model (pytorch model)
    Return:
        number of the parameters with gradients (int)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def _get_arguments():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--waveforms", required=True,
                        type=str, help="directory or list of training wav files")
    parser.add_argument("--feats", required=True,
                        type=str, help="directory or list of aux feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="hdf5 file including statistics")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
    parser.add_argument("--config", required=True,
                        type=str, help="path of the model config")
    # network structure setting
    parser.add_argument("--n_quantize", default=256,
                        type=int, help="number of quantization")
    parser.add_argument("--n_aux", default=39,
                        type=int, help="number of dimensions of aux feats")
    parser.add_argument("--n_resch", default=512,
                        type=int, help="number of channels of residual output")
    parser.add_argument("--n_skipch", default=256,
                        type=int, help="number of channels of skip output")
    parser.add_argument("--dilationF_depth", default=4,
                        type=int, help="depth of fixed dilated network")
    parser.add_argument("--dilationF_repeat", default=3,
                        type=int, help="number of repeating of fixed dilated network")
    parser.add_argument("--dilationA_depth", default=4,
                        type=int, help="depth of adaptive dilated network")
    parser.add_argument("--dilationA_repeat", default=1,
                        type=int, help="number of repeating of adaptive dilated network")
    parser.add_argument("--kernel_size", default=2,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--dense_factor", default=8,
                        type=int, help="number of taps in one cycle")
    parser.add_argument("--upsampling_factor", default=110,
                        type=int, help="upsampling factor of aux features")
    # network training setting
    parser.add_argument("--feature_type", default="world",
                        type=str, help="feature type")
    parser.add_argument("--feature_format", default="h5",
                        type=str, help="feature format")
    parser.add_argument("--batch_length", default=20000,
                        type=int, help="batch length")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="batch size")
    parser.add_argument("--max_length", default=30000,
                        type=int, help="maximum length of batach and receptive field")
    parser.add_argument("--f0_threshold", default=0,
                        type=int, help="threshold of lowest f0")
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="weight decay coefficient")
    parser.add_argument("--iters", default=200000,
                        type=int, help="number of iterations")
    # other setting
    parser.add_argument("--checkpoint_interval", default=10000,
                        type=int, help="how frequent saving model")
    parser.add_argument("--intervals", default=100,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--resume", default=None, nargs="?",
                        type=str, help="model path to restart training")
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
def train_generator(wav_list,
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
                    upsampling_factor=80,
                    shuffle=True):
    """TRAINING BATCH GENERATOR
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
        shuffle (bool): whether to shuffle the file list
    Return:
        (object): generator instance
    """
    # shuffle list
    if shuffle:
        n_files = len(wav_list)
        idx = np.random.permutation(n_files)
        wav_list = [wav_list[i] for i in idx]
        feat_list = [feat_list[i] for i in idx]
    while True:
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
        # re-shuffle
        if shuffle:
            idx = np.random.permutation(n_files)
            wav_list = [wav_list[i] for i in idx]
            feat_list = [feat_list[i] for i in idx]


def _save_checkpoint(checkpoint_dir, model, optimizer, iterations):
    """SAVE CHECKPOINT
    Args:
        checkpoint_dir (str): directory to save checkpoint
        model (torch.nn.Module): pytorch model instance
        optimizer (Optimizer): pytorch optimizer instance
        iterations (int): number of current iterations
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iterations": iterations}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    logging.info("%d-iter checkpoint created." % iterations)


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
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir)

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # save args as conf
    torch.save(args, args.config)
    ## if the program is very slow in some pytorch & cuda version
    ## you can try the following 2 settings
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    # define network
    upsampling_factor = args.upsampling_factor
    if upsampling_factor <= 0:
        logging.warn("upsampling_factor should larger than 0!")
        sys.exit(0)
    model = QPNet(
        n_quantize=args.n_quantize,
        n_aux=args.n_aux,
        n_resch=args.n_resch,
        n_skipch=args.n_skipch,
        dilationF_depth=args.dilationF_depth,
        dilationF_repeat=args.dilationF_repeat,
        dilationA_depth=args.dilationA_depth,
        dilationA_repeat=args.dilationA_repeat,
        kernel_size=args.kernel_size,
        upsampling_factor=upsampling_factor)
    logging.debug(model)
    model.apply(initialize)
    model.train()

    # setups for multi GPUs
    if args.n_gpus > 1:
        device_ids = range(args.n_gpus)
        model = torch.nn.DataParallel(model, device_ids)
        model.receptiveF_field = model.module.receptiveF_field
        model.receptiveA_field = model.module.receptiveA_field
        model.receptiveCausal_field = model.module.receptiveCausal_field
        if args.n_gpus > args.batch_size:
            logging.warn("batch size is less than number of gpus.")

    # define optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # define transforms
    scaler = StandardScaler()
    scaler.mean_ = read_hdf5(args.stats, "/mean")
    scaler.scale_ = read_hdf5(args.stats, "/scale")
    wav_transform = transforms.Compose([
        lambda x: encode_mu_law(x, args.n_quantize)])
    feat_transform = transforms.Compose([
        lambda x: scaler.transform(x)])

    # define generator
    feat_ext = ".%s" % args.feature_format # feature file extension
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
    logging.info("number of training data = %d." % len(wav_list))
    logging.info("max length: %s" % str(args.max_length))
    logging.info("f0 threshold: %s" % str(args.f0_threshold))
    #logging.info("continuous f0: %s" % str(args.contf0_flag))
    assert args.batch_length > 0
    assert args.batch_size > 0    
    generator = train_generator(
        wav_list, feat_list,
        model_receptiveCausal=model.receptiveCausal_field,
        model_receptiveF=model.receptiveF_field,
        model_receptiveA=model.receptiveA_field,
        wav_transform=wav_transform,
        feat_transform=feat_transform,
        feature_type=args.feature_type,
        dense_factor=args.dense_factor,
        batch_length=args.batch_length,
        batch_size=args.batch_size,
        max_length=args.max_length,
        f0_threshold=args.f0_threshold,
        upsampling_factor=upsampling_factor,
        shuffle=True)

    # charge minibatch in queue
    while not generator.queue.full():
        time.sleep(0.1)

    # resume model and optimizer
    flossyml = args.expdir + "/loss-final.yml"
    if os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        iterations = checkpoint["iterations"]
        if args.n_gpus > 1:
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logging.info("restored from %d-iter checkpoint." % iterations)
        if os.path.exists(flossyml):
            with open(flossyml, "r", encoding='utf-8') as yf:
                loss_record = yaml.safe_load(yf)
        else:
            loss_record = []
    else:
        iterations = 0
        loss_record = []

    # check gpu and then send to gpu
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        #sys.exit(1)

    # train
    loss = 0
    total = 0
    logging.info("training start!")
    for i in six.moves.range(iterations, args.iters):
        start = time.time()
        batch_x, batch_h, batch_t, batch_d, batch_b = generator.next()
        batch_output = model(batch_x, batch_h, batch_d, batch_b)

        assert torch.all(batch_b==batch_b[0])
        batch_length_current = int(batch_b[0]) # shift for each layer

        assert(torch.max(batch_t)<args.n_quantize)
        batch_loss = criterion(
            batch_output[:, -batch_length_current:].contiguous().view(-1, args.n_quantize),
            batch_t[:, -batch_length_current:].contiguous().view(-1))
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        total += time.time() - start
        logging.debug("batch loss = %.3f (%.3f sec / batch)" % (
            batch_loss.item(), time.time() - start))

        # report progress
        if (i + 1) % args.intervals == 0:
            logging.info("(iter:%d) average loss = %.6f (%.3f sec / batch)" % (
                i + 1, loss / args.intervals, total / args.intervals))
            logging.info("estimated required time = "
                         "{0.days:02}:{0.hours:02}:{0.minutes:02}:{0.seconds:02}"
                         .format(relativedelta(
                             seconds=int((args.iters - (i + 1)) * (total / args.intervals)))))
            loss_record.append(loss / args.intervals)
            loss = 0
            total = 0

        # save intermidiate model
        if (i + 1) % args.checkpoint_interval == 0:
            if args.n_gpus > 1:
                _save_checkpoint(args.expdir, model.module, optimizer, i + 1)
            else:
                _save_checkpoint(args.expdir, model, optimizer, i + 1)

    # save final model
    if args.n_gpus > 1:
        torch.save({"model": model.module.state_dict()},
                   args.expdir + "/checkpoint-final.pkl")
    else:
        torch.save({"model": model.state_dict()},
                   args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")
    # save the loss record
    with open(args.expdir + "/loss-final.yml", "w", encoding='utf-8') as yf:
        yaml.safe_dump(loss_record, yf)


if __name__ == "__main__":
    main()
