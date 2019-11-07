#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Wu Yi-Chiao (Nagoya University)
# based on a WaveNet script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/PytorchWaveNetVocoder)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import numpy as np
import multiprocessing as mp

def multi_processing(file_list, target_fn, n_jobs):
    # divie list
    file_lists = np.array_split(file_list, n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]
    #print(file_lists[0])
    # multi processing
    processes = []
    for f in file_lists:
        p = mp.Process(target=target_fn, args=(f))
        p.start()
        processes.append(p)
    # wait for all process
    for p in processes:
        p.join()