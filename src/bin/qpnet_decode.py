#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Wu Yi-Chiao (Nagoya University)
# based on a WaveNet script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/PytorchWaveNetVocoder)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import logging
import math
import os
import sys
import copy
import numpy as np
import torch
import torch.multiprocessing as mp

from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from distutils.util import strtobool
from scipy.io import wavfile

from utils import extend_time
from utils import find_files
from utils import read_hdf5
from utils import read_txt
from utils import shape_hdf5

from qpnet import encode_mu_law
from qpnet import decode_mu_law
from qpnet import QPNet


def _get_arguments():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--feats", required=True,
                        type=str, help="list or directory of testing aux feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="hdf5 file including statistics")
    parser.add_argument("--config", required=True,
                        type=str, help="configure file")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    parser.add_argument("--checkpoint", required=True,
                        type=str, help="model file")
    parser.add_argument("--fs", default=22050,
                        type=int, help="sampling rate")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="number of batch size in decoding")
    # other setting
    parser.add_argument("--extra_memory", default=False,
                        type=strtobool, 
                        help=" set True will accelerate the decoding but consume more memory")
    parser.add_argument("--intervals", default=1000,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=100,
                        type=int, help="seed number")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    # f0 scaled setting
    parser.add_argument("--f0_factor", default=1.0,
                        type=float, help="f0 scaled factor")
    parser.add_argument("--f0_dim_index", default=1,
                        type=int, help="f0 dimension index")
    return parser.parse_args()

def pad_list(batch_list, pad_value=0.0):
    """PAD VALUE
    Args:
        batch_list (list): list of batch, where the shape of i-th batch (T_i, C)
        pad_value (float): value to pad
    Return:
        (ndarray): padded batch with the shape (B, T_max, C)
    """
    batch_size = len(batch_list)
    maxlen = max([batch.shape[0] for batch in batch_list])
    n_feats = batch_list[0].shape[-1]
    batch_pad = np.zeros((batch_size, maxlen, n_feats))
    for idx, batch in enumerate(batch_list):
        batch_pad[idx, :batch.shape[0]] = batch

    return batch_pad

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

def _batch_f0(h):
    """LOAD F0 SEQUENCE
    Args:
        h (numpy array): the auxiliary acoustic features (T x D)
    Return:
        cont_f0_lpf(numpy array): 
            float array of the continuous pitch sequence (T)
    """
    #uv = h[:, 0].copy(order='C')  # voive/unvoice feature
    cont_f0_lpf = h[:, 1].copy(order='C')  # continuous f0
    #mcep = h[:, 2:feat_param['mcep_dim_end']].copy(order='C')  # mcc
    #codeap = h[:, feat_param['codeap_index']:].copy(order='C')  # coded ap
    return cont_f0_lpf

def decode_generator(feat_list,
                     fs,
                     wav_transform=None,
                     feat_transform=None,
                     feature_type="world",
                     feat_ext=".h5",
                     dense_factor=8,
                     batch_size=32,
                     upsampling_factor=80,
                     f0_factor=1.0,
                     f0_dim_index=1,
                     extra_memory=False):
    """DECODE BATCH GENERATOR
    Args:
        feat_list (str): list of feat files
        fs (int): sampling rate
        wav_transform (func): preprocessing function for waveform
        feat_transform (func): preprocessing function for aux feats
        feature_type (str): feature type
        feat_ext (str): feature filename extension
        dense_factor (int): the number of taps in one cycle
        batch_size (int): batch size in decoding
        upsampling_factor (int): upsampling factor
        f0_factor (float): the ratio of scaled f0
        f0_dim_index (int): the dimension index of the f0 feature
        extra_memory(bool): processing dilated factor in tensor format or not
                * tensor mode will accelerate the decoding but consume more memory
    Return:
        (object): generator instance
    """
    # sort with the feature length
    shape_list = [shape_hdf5(f, "/" + feature_type)[0] for f in feat_list]
    idx = np.argsort(shape_list)
    feat_list = [feat_list[i] for i in idx]
    
    # divide into batch list
    n_batch = math.ceil(len(feat_list) / batch_size)
    batch_lists = np.array_split(feat_list, n_batch)
    batch_lists = [f.tolist() for f in batch_lists]
    
    for batch_list in batch_lists:
        batch_x = []
        batch_h = []
        batch_d = []
        feat_ids = []
        n_samples_list = []
        for featfile in batch_list:
            # make seed waveform and load aux feature
            x = np.zeros((1))
            h = read_hdf5(featfile, "/" + feature_type)
            if f0_factor is not 1.0:
                h[:, f0_dim_index] = h[:, f0_dim_index] * f0_factor
                d = _dilated_factor(_batch_f0(h), fs, dense_factor)
                d = extend_time(np.expand_dims(d, -1), upsampling_factor) # T x 1
    
            # perform pre-processing
            if wav_transform is not None:
                x = wav_transform(x)
            if feat_transform is not None:
                h = feat_transform(h)
    
            # append to list
            batch_x += [x]
            batch_h += [h]
            batch_d += [d]
            feat_ids += [os.path.basename(featfile).replace(feat_ext, "")]
            n_samples_list += [h.shape[0] * upsampling_factor - 1]
    
        # convert list to ndarray
        batch_x = np.stack(batch_x, axis=0)
        batch_h = pad_list(batch_h)
        batch_d = pad_list(batch_d)
        # convert to torch variable
        batch_x = torch.from_numpy(batch_x).long()
        batch_h = torch.from_numpy(batch_h).float().transpose(1, 2)
        if extra_memory:
            batch_d = torch.from_numpy(batch_d).float().squeeze(-1)
        else:
            batch_d = batch_d.squeeze(-1)
        
        # send to cuda
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_h = batch_h.cuda()
            if extra_memory:
                batch_d = batch_d.cuda()
        
        yield feat_ids, batch_x, batch_h, n_samples_list, batch_d


def main():
    # parser arguments
    args = _get_arguments()
    # set log level
    if args.verbose > 0:
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

    # check directory existence
    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # load config
    config = torch.load(args.config)

    # get file list
    feat_ext = ".%s" % config.feature_format # feature file extension
    if os.path.isdir(args.feats):
        feat_list = sorted(find_files(args.feats, "*%s" % feat_ext))
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)

    # prepare the file list for parallel decoding
    feat_lists = np.array_split(feat_list, args.n_gpus)
    feat_lists = [f_list.tolist() for f_list in feat_lists]

    # define transform
    scaler = StandardScaler()
    scaler.mean_ = read_hdf5(args.stats, "/mean")
    scaler.scale_ = read_hdf5(args.stats, "/scale")
    wav_transform = transforms.Compose([
        lambda x: encode_mu_law(x, config.n_quantize)])
    feat_transform = transforms.Compose([
        lambda x: scaler.transform(x)])
    # define gpu decode function
    def _decode(feat_list, gpu):
        # define model and load parameters
        torch.set_grad_enabled(False)
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
        model.load_state_dict(torch.load(
            args.checkpoint,
            map_location=lambda storage,
            loc: storage)["model"])
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
            model.cuda()
        # define generator
        generator = decode_generator(
            feat_list,
            fs=args.fs,
            wav_transform=wav_transform,
            feat_transform=feat_transform,
            feature_type=config.feature_type,
            feat_ext=feat_ext,
            dense_factor=config.dense_factor,
            batch_size=args.batch_size,
            upsampling_factor=upsampling_factor,
            f0_factor=args.f0_factor,
            f0_dim_index=args.f0_dim_index,
            extra_memory=args.extra_memory)

        # decode
        for feat_ids, batch_x, batch_h, n_samples_list, batch_d in generator:
            logging.info("decoding start!")
            samples_list = model.batch_fast_generate(
                batch_x, batch_h, n_samples_list, batch_d, 
                intervals=args.intervals, extra_memory=args.extra_memory)
            for feat_id, samples in zip(feat_ids, samples_list):
                wav = decode_mu_law(samples, config.n_quantize)
                wav_filename = args.outdir.replace("feat_id", feat_id)
                wav = np.clip(np.int16(wav*32768), -32768, 32767)
                wavfile.write(wav_filename, args.fs, wav)
                logging.info("wrote %s." % (wav_filename))
    
    # parallel decode
    processes = []
    for gpu, feat_list in enumerate(feat_lists):
        p = mp.Process(target=_decode, args=(feat_list, gpu,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
