#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Wu Yi-Chiao (Nagoya University)
# based on sprocket-vc script by Kazuhiro Kobayashi (Nagoya University)
# (https://github.com/k2kobayashi/sprocket)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Generate histograms to decide speaker-dependent parameters

"""

import argparse
import os
import sys
import logging
import multiprocessing as mp

import matplotlib
import numpy as np
from scipy.io import wavfile

from sprocket.speech.feature_extractor import FeatureExtractor
from utils import find_files, read_txt, write_hdf5, check_hdf5

matplotlib.use('Agg')  # noqa #isort:skip
import matplotlib.pyplot as plt  # isort:skip


def create_histogram(data, figure_path, range_min=-70, range_max=20,
                     step=10, xlabel='Power [dB]'):
    """Create histogram
    Parameters
    ----------
    data : list,
        List of several data sequences
    figure_path : str,
        Filepath to be output figure
    range_min : int, optional,
        Minimum range for histogram
        Default set to -70
    range_max : int, optional,
        Maximum range for histogram
        Default set to -20
    step : int, optional
        Stap size of label in horizontal axis
        Default set to 10
    xlabel : str, optional
        Label of the horizontal axis
        Default set to 'Power [dB]'

    """

    # plot histgram
    plt.hist(data, bins=200, range=(range_min, range_max),
             density=True, histtype="stepfilled")
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.xticks(np.arange(range_min, range_max, step))

    figure_dir = os.path.dirname(figure_path)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(figure_path)
    plt.close()

def world_feature_extract(wav_list, idx, f0_dict, npow_dict):
    f0s = []
    npows = []
    for f in wav_list:
        # open waveform
        wavf = f.rstrip()
        fs, x = wavfile.read(wavf)
        x = np.array(x, dtype=np.float)
        logging.info("Extract: " + wavf)

        # constract FeatureExtractor class
        feat = FeatureExtractor(analyzer='world', fs=fs)

        # f0 and npow extraction
        f0, _, _ = feat.analyze(x)
        npow = feat.npow()

        f0s.append(f0)
        npows.append(npow)
    f0_dict[idx] = f0s
    npow_dict[idx] = npows

def main():
    parser = argparse.ArgumentParser(
        description='Create histogram for speaker-dependent configure')
    parser.add_argument('--speaker', default=None,
                        type=str, help='Input speaker label')
    parser.add_argument("--waveforms", default=None, 
                        type=str, help="directory or list of filename of input wavfile")
    parser.add_argument('--figure_dir', default=None,
                        type=str, help='Directory for figure output')
    parser.add_argument("--n_jobs", default=10,
                        type=int, help="number of parallel jobs")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S')

    # show argmument
    for key, value in vars(args).items():
        logging.info("%s = %s" % (key, str(value)))

    # read list
    if os.path.isdir(args.waveforms):
        file_list = sorted(find_files(args.waveforms, "*.wav"))
    else:
        file_list = read_txt(args.waveforms)
    logging.info("number of utterances = %d" % len(file_list))
    
    # divie list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    f0s = [[]] * args.n_jobs
    npows = [[]] * args.n_jobs
    processes = []
    target_fn = world_feature_extract
    manager = mp.Manager()
    f0_dict = manager.dict()
    npow_dict = manager.dict()
    for idx, f in enumerate(file_lists):
        p = mp.Process(target=target_fn, args=(f, idx, f0_dict, npow_dict))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    logging.info("extraction finished!")
    f0 =[]
    npow = []
    for i in range(args.n_jobs):
        f0 += f0_dict[i]
        npow += npow_dict[i]


    f0s = np.hstack(f0).flatten()
    npows = np.hstack(npow).flatten()

    # create a histogram to visualize F0 range of the speaker
    f0histogrampath = os.path.join(
        args.figure_dir, args.speaker + '_f0histogram.png')
    create_histogram(f0s, f0histogrampath, range_min=40, range_max=700,
                     step=50, xlabel='Fundamental frequency [Hz]')
    logging.info("save %s"%f0histogrampath)
    # create a histogram to visualize npow range of the speaker
    npowhistogrampath = os.path.join(
        args.figure_dir, args.speaker + '_npowhistogram.png')
    create_histogram(npows, npowhistogrampath, range_min=-70, range_max=20,
                     step=10, xlabel="Frame power [dB]")
    logging.info("save %s"%npowhistogrampath)



if __name__ == '__main__':
    main()
