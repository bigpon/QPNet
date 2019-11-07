#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Wu Yi-Chiao (Nagoya University)
# based on a WaveNet script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/PytorchWaveNetVocoder)
# based on a voice conversion script by Kazuhiro Kobayashi (Nagoya University)
# (https://github.com/k2kobayashi/sprocket)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Feature extraction script

Usage: runFE.py -e EVALLIST 
                [-hri] [-f FS] 
                [-1] [-2] [-3] [-4] SPK

Options:
    -h, --help     Show the help
    -r, --replace  Over write the exist data 
    -i, --inverse  Inverse flag of filter                   
    -f FS          The sampling rate
    -e EVALLIST    The name of the execute list file
    -1, --step1    Execute step1 (f0 & power statistic)
    -2, --step2    Execute step2 (feature extraction / synthesis)
    -3, --step3    Execute step3 (feature statistic)
    -4, --step4    Execute step4 (waveform noise shaping)
    SPK            The name of speaker       
    
"""
import os
import sys
import h5py
import math
import yaml
import copy
import numpy as np
from docopt import docopt
from sprocket.model.f0statistics import F0statistics
from utils.utils          import write_hdf5, check_hdf5, read_hdf5, read_txt
from utils.param_feat     import acoustic_parameter
from utils.multi_process  import multi_processing
from utils.utils_pathlist import _path_initial, _path_check, _templist, _remove_temp_file
from utils.param_path     import ROOT_DIR, PRJ_DIR, COP_DIR, SRC_DIR
# FEATURE FLAG
SAVE_F0 = True
SAVE_AP = False
SAVE_SPC = False
SAVE_NPOW = True
SAVE_EXTEND = False
SAVE_VAD = True
N_JOBS = 20

# MAIN
if __name__ == "__main__":
    args = docopt(__doc__)  # pylint: disable=invalid-name
    os.environ['PYTHONPATH'] = (SRC_DIR + "utils")
    #print(args)
    # STEP CONTRAL
    execute_steps = [False] + [args["--step{}".format(step_index)] for step_index in range(1, 5)]
    if not any(execute_steps):
        raise("Please specify steps with options")
    # ACOUSTIC FEATURE & WAVEFORM SETTING
    feat_format  = "h5"
    shiftms      = 5
    fs           = "22050"
    if args['-f'] is not None:
        fs = args['-f']
    feat_param   = acoustic_parameter(fs, shiftms=shiftms)
    synonym_root = "rootpath"
    # PATH INITIALIZATION
    spk          = args['SPK']
    tempdir      = "%stemp/" % PRJ_DIR
    corpus_dir   = COP_DIR
    stats_dir    = "%sstats/" % (corpus_dir)
    figure_dir   = "%shist/" % (corpus_dir)
    wavs         = "%sscp/%s" % (corpus_dir, args["-e"])
    spkinfof     = "%sconf/pow_f0_dict.yml" % (corpus_dir)
    _path_check([corpus_dir])
    _path_initial([tempdir, figure_dir, stats_dir, os.path.dirname(spkinfof)])
    running_set  = os.path.basename(wavs).split('.')[0].split("-")[-1]
    stats        = "%s%s_stats.%s" % (stats_dir, running_set, feat_format)
    waveforms    = "%swavs_%s.tmp" % (tempdir, spk)
    _templist(wavs, waveforms, "", [synonym_root], [corpus_dir])
    feats        = "%sfeat_%s.tmp" % (tempdir, running_set)
    _templist(waveforms, feats, "", ["wav"], [feat_format])
    
    # F0 & POW STATISTIC 
    if execute_steps[1]:
        cmd = "python ./bin/initialize_speaker.py" + \
            " --speaker "    + spk + \
            " --waveforms "  + waveforms + \
            " --figure_dir " + figure_dir + \
            " --n_jobs "     + str(N_JOBS)
        #print(cmd)
        os.system(cmd)
        print('f0 & power statistics are created, please modify the %s file for the speaker %s.' % (spkinfof, spk))
        if os.path.exists(spkinfof):
            with open(spkinfof,"r") as f:
                spk_dict = yaml.safe_load(f)
            if not (spk in spk_dict):
                spk_dict.update({spk:{'f0_min':40, 'f0_max':800, 'pow_th':-30}})
        else:
            spk_dict = {spk:{'f0_min':40, 'f0_max':800, 'pow_th':-30}}
        with open(spkinfof,"w") as f:
            yaml.safe_dump(spk_dict, f)
        sys.exit(0)
    
    # FEATURE EXTRACTION / SYNTHESIS
    if execute_steps[2]:
        with open(spkinfof,"r") as f:
            spk_dict = yaml.safe_load(f)
        minf0  = spk_dict[spk]['f0_min']
        maxf0  = spk_dict[spk]['f0_max']
        pow_th = spk_dict[spk]['pow_th']
        cmd = "python ./bin/feature_extract.py" + \
            " --waveforms "       + waveforms + \
            " --feature_type "    + str(feat_param.feature_type)+ \
            " --feature_format "  + str(feat_format) + \
            " --fs "              + str(fs) + \
            " --shiftms "         + str(feat_param.shiftms) + \
            " --fftl "            + str(feat_param.fftl) + \
            " --minf0 "           + str(minf0) + \
            " --maxf0 "           + str(maxf0) + \
            " --pow_th "          + str(pow_th) + \
            " --mcep_dim "        + str(feat_param.mcep_dim) + \
            " --mcep_dim_start "  + str(feat_param.mcep_dim_start) + \
            " --mcep_dim_end "    + str(feat_param.mcep_dim_end) + \
            " --mcep_alpha "      + str(feat_param.mcep_alpha) + \
            " --highpass_cutoff " + str(feat_param.highpass_cutoff) + \
            " --f0_dim_idx "      + str(feat_param.f0_dim_idx) + \
            " --ap_dim_idx "      + str(feat_param.ap_dim_idx) + \
            " --save_f0 "         + str(SAVE_F0) + \
            " --save_ap "         + str(SAVE_AP) + \
            " --save_spc "        + str(SAVE_SPC) + \
            " --save_npow "       + str(SAVE_NPOW) + \
            " --save_extended "   + str(SAVE_EXTEND) + \
            " --save_vad "        + str(SAVE_VAD) + \
            " --overwrite "       + str(args['--replace']) + \
            " --inv "             + str(args['--inverse']) + \
            " --n_jobs "          + str(N_JOBS)
        #print(cmd)
        os.system(cmd)

    # FEATURE STATISTIC
    if execute_steps[3]: 
        cmd = "python ./bin/calc_stats.py" + \
            " --features " + feats + \
            " --stats "    + stats
        # print(cmd)
        os.system(cmd)
    
    # NOISE SHAPING
    if execute_steps[4]:
        wavtype = 'ns'
        cmd = "python ./bin/noise_shaping.py" + \
            " --waveforms "      + waveforms + \
            " --feature_format " + str(feat_format) + \
            " --wavtype "        + wavtype + \
            " --stats "          + stats + \
            " --fs "             + str(fs) + \
            " --shiftms "        + str(feat_param.shiftms) + \
            " --fftl "           + str(feat_param.fftl) + \
            " --mcep_dim_start " + str(feat_param.mcep_dim_start) + \
            " --mcep_dim_end "   + str(feat_param.mcep_dim_end) + \
            " --mcep_alpha "     + str(feat_param.mcep_alpha) + \
            " --mag "            + str(feat_param.mag) + \
            " --n_jobs "         + str(N_JOBS) + \
            " --inv true "
        # print(cmd)
        os.system(cmd)

    _remove_temp_file([waveforms, feats])

    
                
