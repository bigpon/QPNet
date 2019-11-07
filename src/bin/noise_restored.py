#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Wu Yi-Chiao (Nagoya University)
# based on a WaveNet script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/PytorchWaveNetVocoder)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import logging
import multiprocessing as mp
import os
import sys

from distutils.util import strtobool

import numpy as np

from scipy.io import wavfile
from sprocket.speech.feature_extractor import FeatureExtractor
from sprocket.speech.synthesizer import Synthesizer

from feature_extract import low_cut_filter
from utils import find_files, read_hdf5, read_txt, check_hdf5

def _get_arguments():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--feats", required=True,
                        type=str, help="list or directory of aux feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="hdf5 file including statistics")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory of noise shaped wav files")
    parser.add_argument("--writedir", required=True,
                        type=str, help="directory to save restored wav file")
    # feature setting
    parser.add_argument("--feature_format", default="h5",
                        type=str, help="feature format")
    parser.add_argument("--pow_adjust", default="1.0",
                        type=float, help="power adjust factor") 
    parser.add_argument("--fs", default=16000,
                        type=int, help="Sampling frequency")
    parser.add_argument("--shiftms", default=5,
                        type=float, help="Frame shift in msec")
    parser.add_argument("--fftl", default=1024,
                        type=int, help="FFT length")
    parser.add_argument("--mcep_dim_start", default=2,
                        type=int, help="Start index of mel cepstrum")
    parser.add_argument("--mcep_dim_end", default=27,
                        type=int, help="End index of mel cepstrum")
    parser.add_argument("--mcep_alpha", default=0.41,
                        type=float, help="Alpha of mel cepstrum")
    parser.add_argument("--mag", default=0.5,
                        type=float, help="magnification of noise shaping")
    # other setting
    parser.add_argument("--verbose", default=1,
                        type=int, help="log message level")
    parser.add_argument('--n_jobs', default=40,
                        type=int, help="number of parallel jobs")
    parser.add_argument('--inv', default=False, 
                        type=strtobool, help="if True, inverse filtering will be performed")

    return parser.parse_args()

def noise_shaping(wav_list, args):
    """APPLY NOISE SHAPING"""
    # define feature extractor
    feature_extractor = FeatureExtractor(
        analyzer="world",
        fs=args.fs,
        shiftms=args.shiftms,
        fftl=args.fftl)

    # define synthesizer
    synthesizer = Synthesizer(
        fs=args.fs,
        shiftms=args.shiftms,
        fftl=args.fftl)

    for i, feat_id in enumerate(wav_list):
        logging.info("now processing %s (%d/%d)" % (feat_id, i + 1, len(wav_list)))
        # load wavfile and apply low cut filter
        wav_filename = args.outdir.replace("feat_id", feat_id)
        fs, x = wavfile.read(wav_filename)
        wav_type = x.dtype
        x = np.array(x, dtype=np.float64)

        # check sampling frequency
        if not fs == args.fs:
            logging.error("sampling frequency is not matched.")
            sys.exit(1)

        ## extract features (only for get the number of frames)
        f0, _, _ = feature_extractor.analyze(x)
        num_frames = f0.shape[0]

        # load average mcep
        mlsa_coef = read_hdf5(args.stats, "/mean")
        mlsa_coef = mlsa_coef[args.mcep_dim_start:args.mcep_dim_end] * args.mag
        mlsa_coef[0] = 0.0
        if args.inv:
            mlsa_coef[1:] = -1.0 * mlsa_coef[1:]
        mlsa_coef = np.tile(mlsa_coef, [num_frames, 1])
        
        # synthesis and write
        x_ns = synthesizer.synthesis_diff(x, mlsa_coef, alpha=args.mcep_alpha)
        x_ns = low_cut_filter(x_ns, args.fs, cutoff=70)
        write_name = args.writedir.replace("feat_id", feat_id)
        # check directory existence
        if wav_type == np.int16:
            wav = np.clip(np.int16(x_ns), -32768, 32767)
        else:
            wav = np.clip(x_ns, -32768, 32767)
        wavfile.write(write_name, args.fs, wav)


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

    # check directory existence
    if not os.path.exists(os.path.dirname(args.writedir)):
        os.makedirs(os.path.dirname(args.writedir))

    # get file list
    if os.path.isdir(args.feats):
        feat_list = sorted(find_files(args.feats, "*.%s" % args.feature_format))
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)
    feat_ids = [os.path.basename(f).replace(".%s" % args.feature_format, "") for f in feat_list]
    logging.info("number of utterances = %d" % len(feat_ids))

    # divie list
    feat_ids = np.array_split(feat_ids, args.n_jobs)
    feat_ids = [f_ids.tolist() for f_ids in feat_ids]

    # multi processing
    processes = []
    # for f in file_lists:
    for f in feat_ids:
        p = mp.Process(target=noise_shaping, args=(f, args,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
