#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Wu Yi-Chiao (Nagoya University)
# based on a WaveNet script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/PytorchWaveNetVocoder)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Pytorch QPNet script

Usage: runQP.py -w WAVLIST -a AUXLIST
                [-hmr] [-f FS]
                [-x UPWAVLIST] [-u UPAUXLIST]
                [-y VALIDWAVLIST] [-v VALIDAUXLIST] 
                [-e EVALLIST]
                [-g GPUID] [-n NETWORK] [-d DENSE]              
                [-I ITER] [-U UITER] 
                [-R RESUME] [-M MODEL]
                [-1] [-2] [-3] [-4] [-5] [TESTSPK]
              
Options:
    -h, --help       Show the help
    -r, --replace    Over write the exist evaluation results
    -m, --multi      Multi-speaker QPNet generatiron
    -w WAVLIST       The list of the training waveform files
    -a AUXLIST       The list of the training auxiliary features
    -x UPWAVLIST     The list of the updating waveform files
    -u UPAUXLIST     The list of the updating auxiliary features
    -y VALIDWAVLIST  The list of the validation waveform files
    -v VALIDAUXLIST  The list of the validation auxiliary features
    -e EVALLIST      The list of the evaluation features
    -f FS            The sampling rate
    -g GPUID         The GPU device ID
    -n NETWORK       The name of the network structure ('d4r4')
    -d DENSE         The dense factor a
    -I ITER          The number of iteration
    -U UITER         The number if update iteration
    -R RESUME        The number of iteration to resume model
    -M MODEL         The number of iteration of model for testing
    -1, --step1      Execute step1 (train QPNet)
    -2, --step2      Execute step2 (update QPNet)
    -3, --step3      Execute step3 (QPNet decode)
    -4, --step4      Execute step4 (noiseshaping restored)
    -5, --step5      Execute step5 (validation)
    TESTSPK          The speaker name of the evaluation list
    
"""
import os
import sys
import h5py
import math
import yaml
import numpy as np
from docopt import docopt
from utils.utils_pathlist import _path_initial, _path_check, _list_initial, _remove_temp_file
from utils.utils_pathlist import _templist, _templist_eval
from utils.param_model    import qpwn_parameter
from utils.param_feat     import acoustic_parameter
from utils.param_path     import LIBRARY_DIR, CUDA_DIR, ROOT_DIR 
from utils.param_path     import PRJ_DIR, COP, COP_DIR, SCP_DIR, SRC_DIR
N_JOBS = 25
N_GPUS = 1
SEED   = 1
DECODE_SEED       = 100
DECODE_BATCH_SIZE = 20

# MAIN
if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    # STEP CONTRAL
    execute_steps = [False] \
        + [args["--step{}".format(step_index)] for step_index in range(1, 6)]
    if not any(execute_steps):
        raise("Please specify steps with options")
    # ENVIRONMET PARAMETER SETTING
    os.environ['LD_LIBRARY_PATH'] += ":" + LIBRARY_DIR
    os.environ['CUDA_HOME'] = CUDA_DIR
    os.environ['PATH'] += (":" + SRC_DIR + "bin:" + SRC_DIR + "utils")
    os.environ['PYTHONPATH'] = (SRC_DIR + "utils")
    os.environ['PYTHONPATH'] += (":" + SRC_DIR + "nets")
    os.environ["CUDA_DEVICE_ORDER"]	= "PCI_BUS_ID"
    if args['-g'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['-g']
        num_gpus = 1
    else:
        num_gpus = N_GPUS
    # ACOUSTIC FEATURE & WAVEFORM SETTING
    feat_format     = "h5"
    shiftms         = 5
    wav_mode        = "noiseshaped"
    synonym_wavtype = "wav_%s_ns" % (feat_format)
    filter_version  = "noise_restored"
    mag             = 0.5
    pow_adjust      = 1.0
    fs              = "22050"
    restored_mode   = "restored"
    if args['-f'] is not None:
        fs = args['-f']
    feat_param = acoustic_parameter(fs, shiftms=shiftms)
    # RUNNING SETTING
    network            = "qpnet"
    synonym_root       = "rootpath"
    execution_root     = "./bin"
    execution_train    = "%s/%s_train.py"    % (execution_root, network)
    execution_update   = "%s/%s_update.py"   % (execution_root, network) 
    execution_validate = "%s/%s_validate.py" % (execution_root, network) 
    execution_decode   = "%s/%s_decode.py"   % (execution_root, network)
    execution_filter   = "%s/%s.py" % (execution_root, filter_version)
    # MODEL SETTING
    if args['-d'] is not None:
        dense_factor = np.int(args['-d'])
    else:
        dense_factor = 8
    aux_version     = os.path.basename(args['-a']).split(".")[0].split("-")[-1]
    wav_version     = os.path.basename(args['-w']).split(".")[0].split("-")[-1]
    model_version   = "A%s_W%s_d%d" % (aux_version, wav_version, dense_factor) # model name    
    net_name        = "default" # network structure
    iters           = "200000"  # number of training iteration
    check_interval  = "10000"
    up_iters        = "3000"   # number of updating iteration
    update_interval = "100"
    model_iters     = "final"   # testing model version
    if args['-n'] is not None:
        net_name = args['-n']
        if net_name != "default":
            model_version   = "%s_%s" % (model_version, net_name)
    if args['-I'] is not None:
        iters = args['-I']
    if args['-U'] is not None:
        up_iters = args['-U']
    if args['-M'] is not None:
        model_iters = args['-M']
    model_param = qpwn_parameter(net_name, 
                                 aux=int(feat_param.aux_dim),
                                 iters=int(iters), 
                                 update_iters= int(up_iters),
                                 checkpoint_interval=int(check_interval), 
                                 update_interval=int(update_interval),
                                 decode_batch_size=DECODE_BATCH_SIZE)
    validation_intervel = range(model_param.checkpoint_interval,
                                model_param.iters+1,
                                model_param.checkpoint_interval)
    # PATH INITIALIZATION
    corpus_dir    = COP_DIR
    scp_dir       = SCP_DIR
    stats         = "%sstats/%s_stats.%s" % (corpus_dir, wav_version, feat_format)
    expdir        = "%s%s_models/%s/" % (PRJ_DIR, network, model_version)
    outdir        = "%s%s_output/%s/" % (PRJ_DIR, network, model_version)
    config        = expdir + "model.conf"
    tempdir       = "%stemp/" % PRJ_DIR
    _path_initial([tempdir])    
    _path_check([corpus_dir, stats])
    #LIST INITIALIZATION 
    def _get_list(auxlist, wavlist, modelver, setname):
        # get auxiliary feat list
        aux_feats = "%s%s%s_%sauxfeats.tmp" % (tempdir, COP, modelver, setname)
        _templist(auxlist, aux_feats, "", [synonym_root, "wav"], [corpus_dir, feat_format])
        # get waveform list
        waveforms = "%s%s%s_%swaveforms.tmp" % (tempdir, COP, modelver, setname)
        keyword = [synonym_root, "wav", ".%s"%synonym_wavtype]
        subword = [corpus_dir, synonym_wavtype, ".wav"]
        _templist(wavlist, waveforms, "", keyword, subword)
        return aux_feats, waveforms
    # get training auxiliary feat & waveform list
    aux_feats, waveforms = _get_list(scp_dir + args['-a'], 
                                     scp_dir + args['-w'], 
                                     model_version, 'training')
    
    # NETWORK TRAINING
    if execute_steps[1]:
        # resume setting
        if args['-R'] is not None:
            resume = expdir + "checkpoint-%s.pkl" % (args['-R'])
            _path_check([resume])
        else:
            resume = "None"
        # training
        cmd = "python "                + execution_train + \
            " --waveforms "            + waveforms + \
            " --feats "                + aux_feats + \
            " --stats "                + stats + \
            " --expdir "               + expdir + \
            " --config "               + config + \
            " --n_quantize "           + str(model_param.quantize) + \
            " --n_aux "                + str(model_param.aux) + \
            " --n_resch "              + str(model_param.resch) + \
            " --n_skipch "             + str(model_param.skipch) + \
            " --dilationF_depth "      + str(model_param.dilationF_depth) + \
            " --dilationF_repeat "     + str(model_param.dilationF_repeat) + \
            " --dilationA_depth "      + str(model_param.dilationA_depth) + \
            " --dilationA_repeat "     + str(model_param.dilationA_repeat) + \
            " --kernel_size "          + str(model_param.kernel_size) + \
            " --dense_factor "         + str(dense_factor) + \
            " --upsampling_factor "    + str(feat_param.upsampling_factor)+ \
            " --feature_type "         + str(feat_param.feature_type)+ \
            " --feature_format "       + str(feat_format) + \
            " --batch_length "         + str(model_param.batch_length) + \
            " --batch_size "           + str(model_param.batch_size) + \
            " --max_length "           + str(model_param.max_length) + \
            " --f0_threshold "         + str(model_param.f0_threshold) + \
            " --lr "                   + str(model_param.lr) + \
            " --weight_decay "         + str(model_param.weight_decay) + \
            " --iters "                + str(model_param.iters) + \
            " --checkpoint_interval "  + str(model_param.checkpoint_interval) + \
            " --seed "                 + str(SEED) + \
            " --resume "               + resume + \
            " --n_gpus "               + str(num_gpus) + \
            " --verbose 1 "
        #print(cmd)
        os.system(cmd)
        _remove_temp_file([waveforms, aux_feats])
    
    
    # NETWORK ADAPTATION
    if not args['--multi']:
        if args['-u'] is None or args['-x'] is None:
            print("Please assign the updating auxilary list by '-u UPAUXLIST' " + \
                  " and the corresponding wav list by '-x UPWAVLIST' " + \
                  "or select the multi speaker mode by '--multi'.")
            sys.exit(0)
        # check the pretrained checkpoint
        pretrain_checkpoint = "%s/checkpoint-final.pkl" % (expdir)
        _path_check([pretrain_checkpoint])
        # get updating model version
        upaux_version = os.path.basename(args['-u']).split(".")[0].split("-")[-1]
        upwav_version = os.path.basename(args['-x']).split(".")[0].split("-")[-1]
        model_version = "%s_U%s_V%s" % (model_version, upaux_version, upwav_version)
        # get updating auxiliary feat & waveform list
        upaux_feats, upwaveforms = _get_list(scp_dir + args['-u'], 
                                             scp_dir + args['-x'], 
                                             model_version, 'updating')
        # update path
        expdir = "%s%s_models/%s/" % (PRJ_DIR, network, model_version)
        outdir = "%s%s_output/%s/" % (PRJ_DIR, network, model_version)
        # update validation interval
        validation_intervel = range(model_param.update_interval,
                                    model_param.update_iters+1,
                                    model_param.update_interval)
        # resume setting
        if args['-R'] is not None:
            resume = expdir + "checkpoint-%s.pkl" % (args['-R'])
            _path_check([resume])
        else:
            resume = "None"
        # adaptation
        if execute_steps[2]:
            cmd = "python "                + execution_update + \
                " --waveforms "            + upwaveforms + \
                " --feats "                + upaux_feats + \
                " --stats "                + stats + \
                " --expdir "               + expdir + \
                " --config "               + config + \
                " --pretrain "             + pretrain_checkpoint + \
                " --batch_length "         + str(model_param.batch_length) + \
                " --batch_size "           + str(model_param.batch_size) + \
                " --max_length "           + str(model_param.max_length) + \
                " --f0_threshold "         + str(model_param.f0_threshold) + \
                " --lr "                   + str(model_param.lr) + \
                " --weight_decay "         + str(model_param.weight_decay) + \
                " --iters "                + str(model_param.update_iters) + \
                " --checkpoint_interval "  + str(model_param.update_interval) + \
                " --resume "               + resume + \
                " --seed "                 + str(SEED) + \
                " --n_gpus "               + str(num_gpus) + \
                " --verbose 1 "
            # print(cmd)
            os.system(cmd)
        _remove_temp_file([upwaveforms, upaux_feats])

    # EVALUATION
    if args['-e'] is None:
        print("(warning) test list is empty.")
    else:
        # testing settings initialization
        if args['TESTSPK'] is None:
            print("Pleas assign the evaluation speaker.")
            sys.exit(0)
        testspk      = args['TESTSPK']
        outdir_eval  = os.path.join(outdir, wav_mode, testspk, model_iters, "feat_id.wav")
        test_feats   = "%s%s%s_testfeats.tmp" % (tempdir, COP, model_version)
        tlist        = scp_dir + args['-e']
        keyword      = [synonym_root, "wav"]
        subword      = [corpus_dir, feat_format]
        f0_factor    = 1.0 # f0 scaled factor (1.0 means unchanged)
        extra_memory = False # set True will accelerate the decoding but consume lots of memory
        # speech decoding
        if execute_steps[3]:
            final_checkpoint = "%s/checkpoint-%s.pkl" % (expdir, model_iters)
            _path_check([final_checkpoint])
            # check the evaluation list 
            if not _list_initial(args['--replace'], feat_format, tlist, test_feats, outdir_eval, keyword, subword):
                print("%s is skipped" % (args['-e']))
            else:
                cmd = "python "          + execution_decode + \
                    " --feats "          + test_feats + \
                    " --stats "          + stats + \
                    " --config "         + config + \
                    " --outdir "         + outdir_eval + \
                    " --checkpoint "     + final_checkpoint + \
                    " --fs "             + str(feat_param.fs) + \
                    " --batch_size "     + str(model_param.decode_batch_size) + \
                    " --extra_memory "   + str(extra_memory) + \
                    " --seed "           + str(DECODE_SEED) + \
                    " --n_gpus "         + str(num_gpus) + \
                    " --f0_factor "      + str(f0_factor) + \
                    " --f0_dim_index "   + str(feat_param.f0_dim_idx)
                # print(cmd)
                os.system(cmd)

        # noise shaping restored
        if execute_steps[4]:
            _path_check([os.path.dirname(outdir_eval)])
            writedir = outdir_eval.replace(wav_mode, restored_mode)
            _templist(tlist, test_feats, outdir_eval, keyword, subword)
            cmd = "python "          + execution_filter + \
                " --feats "          + test_feats + \
                " --stats "          + stats + \
                " --outdir "         + outdir_eval + \
                " --writedir "       + writedir + \
                " --feature_format " + feat_format + \
                " --pow_adjust "     + str(pow_adjust) + \
                " --fs "             + str(feat_param.fs) + \
                " --shiftms "        + str(feat_param.shiftms) + \
                " --fftl "           + str(feat_param.fftl) + \
                " --mcep_dim_start " + str(feat_param.mcep_dim_start) + \
                " --mcep_dim_end "   + str(feat_param.mcep_dim_end) + \
                " --mcep_alpha "     + str(feat_param.mcep_alpha) + \
                " --mag "            + str(mag) + \
                " --n_jobs "         + str(N_JOBS) + \
                " --inv false"
            # print(cmd)
            os.system(cmd)
        _remove_temp_file([test_feats])
    
    # NETWORK VALIDATION
    if execute_steps[5]:
        if args['-v'] is None or args['-y'] is None:
            print("Please assign the validation auxilary list by '-v VALIDAUXLIST' " + \
                  " and the corresponding wav list by '-y VALIDWAVLIST' ")
            sys.exit(0)
        # get validation auxiliary feat & waveform list
        validaux_feats, validwaveforms = _get_list(scp_dir + args['-v'], 
                                                   scp_dir + args['-y'], 
                                                   model_version, 'validation')
        for model_iters in validation_intervel:
            checkpoint = "%s/checkpoint-%s.pkl" % (expdir, model_iters)
            _path_check([checkpoint])
            cmd = "python "        + execution_validate + \
                " --waveforms "    + validwaveforms + \
                " --feats "        + validaux_feats + \
                " --stats "        + stats + \
                " --resultdir "    + expdir + \
                " --config "       + config + \
                " --checkpoint "   + checkpoint + \
                " --batch_length " + str(model_param.batch_length) + \
                " --batch_size "   + str(model_param.batch_size) + \
                " --max_length "   + str(model_param.max_length) + \
                " --n_gpus "       + str(num_gpus) + \
                " --verbose 1 "
            # print(cmd)
            os.system(cmd)
        _remove_temp_file([validwaveforms, validaux_feats])