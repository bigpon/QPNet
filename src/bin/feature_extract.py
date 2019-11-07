#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Wu Yi-Chiao (Nagoya University)
# based on a WaveNet script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/PytorchWaveNetVocoder)
# based on sprocket-vc script by Kazuhiro Kobayashi (Nagoya University)
# (https://github.com/k2kobayashi/sprocket)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import logging
import multiprocessing as mp
import os
import sys
import copy
import pyworld
import numpy as np

from distutils.util import strtobool
from numpy.matlib import repmat
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import firwin
from scipy.signal import lfilter
from sprocket.speech.feature_extractor import FeatureExtractor
from sprocket.speech.synthesizer import Synthesizer
from utils import (find_files, read_txt, read_hdf5, write_hdf5, check_hdf5)

def _get_arguments():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")
    # path setting
    parser.add_argument("--waveforms", required=True,
                        type=str, help="directory or list of input wav files")
    parser.add_argument("--feature_dir", default=None, 
                        type=str, help="directory of output featfile")
    # acoustic feature setting
    parser.add_argument("--feature_type", default="world", choices=["world"],
                        type=str, help="feature type")
    parser.add_argument("--feature_format", default="h5",
                        type=str, help="feature format")
    parser.add_argument("--fs", default=22050,
                        type=int, help="sampling frequency")
    parser.add_argument("--shiftms", default=5.0,
                        type=float, help="frame shift in msec")
    parser.add_argument("--fftl", default=1024,
                        type=int, help="FFT length")
    parser.add_argument("--minf0", default=40,
                        type=float, help="minimum f0")
    parser.add_argument("--maxf0", default=400,
                        type=float, help="maximum f0")
    parser.add_argument("--pow_th", default=-20,
                        type=float, help="speech power threshold")                        
    parser.add_argument("--mcep_dim", default=34,
                        type=int, help="dimension of mel cepstrum")
    parser.add_argument("--mcep_dim_start", default=2,
                        type=int, help="first dimension index of mel cepstrum")
    parser.add_argument("--mcep_dim_end", default=37,
                        type=int, help="last dimension index of mel cepstrum")
    parser.add_argument("--mcep_alpha", default=0.455,
                        type=float, help="Alpha of mel cepstrum")
    parser.add_argument("--highpass_cutoff", default=70,
                        type=int, help="cut off frequency in lowpass filter")
    parser.add_argument("--f0_dim_idx", default=1,
                        type=int, help="f0 dimension index")
    parser.add_argument("--ap_dim_idx", default=-2,
                        type=int, help="ap dimension index")
    # flags setting
    parser.add_argument("--save_f0", default=True,
                        type=strtobool, help="if set True, features f0 will be saved")
    parser.add_argument("--save_ap", default=False,
                        type=strtobool, help="if set True, features ap will be saved")
    parser.add_argument("--save_spc", default=False,
                        type=strtobool, help="if set True, features spc will be saved")
    parser.add_argument("--save_npow", default=True,
                        type=strtobool, help="if set True, features npow will be saved")
    parser.add_argument("--save_extended", default=False,
                        type=strtobool, help="if set True, exteneded feature will be saved")
    parser.add_argument("--save_vad", default=True,
                        type=strtobool, help="if set True, features vad_idx will be saved")
    parser.add_argument("--overwrite", default=False,
                        type=strtobool, help="if set True, overwrite the exist feature files")
    # other setting
    parser.add_argument('--inv', default=True, 
                        type=strtobool, help="if False, wav is restored from acoustic features")
    parser.add_argument("--n_jobs", default=10,
                        type=int, help="number of parallel jobs")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log message level")

    return parser.parse_args()

def rootdir_replace(filepath, extname=None, newdir=None):
    filename = os.path.basename(filepath)
    rootdir  = os.path.dirname(filepath)
    if extname != None:
        filename = '%s.%s' % (filename.split('.')[0], extname)
    if newdir == None:
        newdir = rootdir
    return '%s/%s'%(newdir, filename)

def extfrm(data, npow, power_threshold=-20):
    T = data.shape[0]
    if T != len(npow):
        raise("Length of two vectors is different.")

    valid_index = np.where(npow > power_threshold)
    extdata = data[valid_index]
    assert extdata.shape[0] <= T

    return extdata, valid_index[0]

def low_cut_filter(x, fs, cutoff=70):
    """FUNCTION TO APPLY LOW CUT FILTER
    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter
    Return:
        (ndarray): Low cut filtered waveform sequence
    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x

def low_pass_filter(x, fs, cutoff=70, padding=True):
    """APPLY LOW PASS FILTER
    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter
    Return:
        (ndarray): Low pass filtered waveform sequence
    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    # low cut filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

    return lpf_x

def extend_time(feats, upsampling_factor):
    """EXTEND TIME RESOLUTION
    Args:
        feats (ndarray): feature vector with the shape (T x D)
        upsampling_factor (int): upsampling_factor
    Return:
        (ndarray): extend feats with the shape (upsampling_factor*T x D)
    """
    # get number
    n_frames = feats.shape[0]
    n_dims = feats.shape[1]
    # extend time
    feats_extended = np.zeros((n_frames * upsampling_factor, n_dims))
    for j in range(n_frames):
        start_idx = j * upsampling_factor
        end_idx = (j + 1) * upsampling_factor
        feats_extended[start_idx: end_idx] = repmat(feats[j, :], upsampling_factor, 1)

    return feats_extended

def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0
    Args:
        f0 (ndarray): original f0 sequence with the shape (T)
    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)
    # get start and end of f0
    if (f0 == 0).all():
        logging.warn("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    # padding start and end of f0 sequence
    cont_f0 = copy.deepcopy(f0)
    start_idx = np.where(cont_f0 == start_f0)[0][0]
    end_idx = np.where(cont_f0 == end_f0)[0][-1]
    cont_f0[:start_idx] = start_f0
    cont_f0[end_idx:] = end_f0
    # get non-zero frame index
    nz_frames = np.where(cont_f0 != 0)[0]
    # perform linear interpolation
    f = interp1d(nz_frames, cont_f0[nz_frames])
    cont_f0 = f(np.arange(0, cont_f0.shape[0]))
    return uv, cont_f0

def featpath_create(wav_list, feature_format):
    """CREATE FILE FOLDER"""
    for wav_name in wav_list:
        feat_name = wav_name.replace("wav", feature_format)
        if not os.path.exists(os.path.dirname(feat_name)):
            os.makedirs(os.path.dirname(feat_name))

def wavpath_create(wav_list, feature_format):
    """CREATE FILE FOLDER"""
    for wav_name in wav_list:
        restored_name = wav_name.replace("wav", feature_format+"_restored")
        if not os.path.exists(os.path.dirname(restored_name)):
            os.makedirs(os.path.dirname(restored_name))

def world_speech_synthesis(queue, wav_list, args):
    """WORLD SPEECH SYNTHESIS
    Parameters
    ----------
    queue : multiprocessing.Queue()
        the queue to store the file name of utterance
    wav_list : list
        list of the wav files
    args : 
        feature extract arguments
    """
    # define ynthesizer
    synthesizer = Synthesizer(fs=args.fs,
                              fftl=args.fftl,
                              shiftms=args.shiftms)
    # synthesis
    for i, wav_name in enumerate(wav_list):
        if args.feature_dir==None:
            restored_name = wav_name.replace("wav", args.feature_format+"_restored")
            restored_name = restored_name.replace(".%s" % args.feature_format+"_restored", ".wav")
            feat_name = wav_name.replace("wav", args.feature_format)
        else:
            restored_name = rootdir_replace(wav_name, newdir=args.feature_dir+"restored")
            feat_name = rootdir_replace(wav_name, extname=args.feature_format, newdir=args.feature_dir)
        if os.path.exists(restored_name):
            if args.overwrite:
                logging.info("overwrite %s (%d/%d)" % (restored_name, i + 1, len(wav_list)))
            else:
                logging.info("skip %s (%d/%d)" % (restored_name, i + 1, len(wav_list)))            
                continue
        else:
            logging.info("now processing %s (%d/%d)" % (restored_name, i + 1, len(wav_list)))
        # load acoustic features
        if check_hdf5(feat_name, "/world"):
            h = read_hdf5(feat_name, "/world")
        else:
            logging.error("%s is not existed."%(feat_name))
            sys.exit(1)
        if check_hdf5(feat_name, "/f0"):
            f0 = read_hdf5(feat_name, "/f0")
        else:
            uv = h[:, 0].copy(order='C')
            f0 = h[:, args.f0_dim_idx].copy(order='C')  # cont_f0_lpf
            fz_idx = np.where(uv==0.0)
            f0[fz_idx] = 0.0
        if check_hdf5(feat_name, "/ap"):
            ap = read_hdf5(feat_name, "/ap")
        else:
            codeap = h[:, args.ap_dim_idx:].copy(order='C')
            ap = pyworld.decode_aperiodicity(codeap, args.fs, args.fftl)
        mcep = h[:, args.mcep_dim_start:args.mcep_dim_end].copy(order='C')
        # waveform synthesis
        wav = synthesizer.synthesis(f0,
                                    mcep,
                                    ap,
                                    alpha=args.mcep_alpha)
        wav = np.clip(np.int16(wav), -32768, 32767)
        wavfile.write(restored_name, args.fs, wav)
        #logging.info("wrote %s." % (restored_name))
    queue.put('Finish')

def world_feature_extract(queue, wav_list, args):
    """EXTRACT WORLD FEATURE VECTOR
    Parameters
    ----------
    queue : multiprocessing.Queue()
        the queue to store the file name of utterance
    wav_list : list
        list of the wav files
    args : 
        feature extract arguments
    """
    # define feature extractor
    feature_extractor = FeatureExtractor(
        analyzer="world",
        fs=args.fs,
        shiftms=args.shiftms,
        minf0=args.minf0,
        maxf0=args.maxf0,
        fftl=args.fftl)
    # extraction
    for i, wav_name in enumerate(wav_list):
        # check exists
        if args.feature_dir==None:
            feat_name = wav_name.replace("wav", args.feature_format)
        else:
            feat_name = rootdir_replace(wav_name, extname=args.feature_format, newdir=args.feature_dir)
        #if not os.path.exists(os.path.dirname(feat_name)):
        #    os.makedirs(os.path.dirname(feat_name))
        if check_hdf5(feat_name, "/world"):
            if args.overwrite:
                logging.info("overwrite %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))
            else:
                logging.info("skip %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))            
                continue
        else:
            logging.info("now processing %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))
        # load wavfile and apply low cut filter
        fs, x = wavfile.read(wav_name)
        x = np.array(x, dtype=np.float32)
        if args.highpass_cutoff != 0:
            x = low_cut_filter(x, fs, cutoff=args.highpass_cutoff)

        # check sampling frequency
        if not fs == args.fs:
            logging.error("sampling frequency is not matched.")
            sys.exit(1)

        # extract features
        f0, spc, ap = feature_extractor.analyze(x)
        codeap = feature_extractor.codeap()
        mcep = feature_extractor.mcep(dim=args.mcep_dim, alpha=args.mcep_alpha)
        npow = feature_extractor.npow()
        uv, cont_f0 = convert_continuos_f0(f0)
        lpf_fs = int(1.0 / (args.shiftms * 0.001))
        cont_f0_lpf = low_pass_filter(cont_f0, lpf_fs, cutoff=20)
        next_cutoff = 70
        while not (cont_f0_lpf>[0]).all():
            logging.info("%s low-pass-filtered [%dHz]" % (feat_name, next_cutoff))
            cont_f0_lpf = low_pass_filter(cont_f0, lpf_fs, cutoff=next_cutoff)
            next_cutoff *= 2

        # concatenate
        cont_f0_lpf = np.expand_dims(cont_f0_lpf, axis=-1)
        uv = np.expand_dims(uv, axis=-1)
        feats = np.concatenate([uv, cont_f0_lpf, mcep, codeap], axis=1)
        
        # save feature
        write_hdf5(feat_name, "/world", feats)
        if args.save_f0:
            write_hdf5(feat_name, "/f0", f0)
        if args.save_ap:
            write_hdf5(feat_name, "/ap", ap)
        if args.save_spc:
            write_hdf5(feat_name, "/spc", spc)
        if args.save_npow:
            write_hdf5(feat_name, "/npow", npow)
        if args.save_extended:
            # extend time resolution
            upsampling_factor = int(args.shiftms * fs * 0.001)
            feats_extended = extend_time(feats, upsampling_factor)
            feats_extended = feats_extended.astype(np.float32)
            write_hdf5(feat_name, "/world_extend", feats_extended)
        if args.save_vad:
            _, vad_idx = extfrm(mcep, npow, power_threshold=args.pow_th)
            write_hdf5(feat_name, "/vad_idx", vad_idx)
    queue.put('Finish')


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

    # read list
    if os.path.isdir(args.waveforms):
        file_list = sorted(find_files(args.waveforms, "*.wav"))
    else:
        file_list = read_txt(args.waveforms)
    logging.info("number of utterances = %d" % len(file_list))

    # set mode
    if args.feature_type == "world":
        if args.inv:
            target_fn = world_feature_extract
            filepath_create = featpath_create
        else:
            target_fn = world_speech_synthesis
            filepath_create = wavpath_create        
    else:
        raise NotImplementedError("Currently, only support  world.")

    # create file folders
    if args.feature_dir==None:
        filepath_create(file_list, args.feature_format)
    else:
        featdir = args.feature_dir
        if not os.path.exists(featdir):
            os.makedirs(featdir)
        if not os.path.exists(featdir+"restored/"):
            os.makedirs(featdir+"restored/")

    # divide list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    processes = []
    queue = mp.Queue()
    for f in file_lists:
        p = mp.Process(target=target_fn, args=(queue, f, args,))
        p.start()
        processes.append(p)
    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
