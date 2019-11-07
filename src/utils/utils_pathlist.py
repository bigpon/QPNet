#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Wu Yi-Chiao (Nagoya University)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os

# PATH INITIALIZATION
def _path_initial(pathlist):
    for pathdir in pathlist:
        if not os.path.exists(pathdir):
            os.makedirs(pathdir)

# PATH CHECK
def _path_check(pathlist):
    for pathdir in pathlist:
        if not os.path.exists(pathdir):
            raise FileNotFoundError("%s doesn't exist!!" % pathdir)
        
# LIST INITIALIZATION
def _list_initial(replace_falg, feat_format, wavs, feats, outdir, keyword, subword):
    if replace_falg:
        _templist_eval(wavs, feats, outdir, overwrite=True,
                       rootdir="", keyword=keyword, subword=subword)
    else:
        num_files = _templist_eval(wavs, feats, outdir, overwrite=False,
                                   rootdir="", keyword=keyword, 
                                   subword=subword, feat_format=feat_format)
        if num_files is 0:
            return False
    return True
            
# TRAINING LIST PROCESSING
def _templist(inputlist, outputlist,
             rootdir="", keyword="", subword=""):
    num_files = 0
    if not os.path.exists(inputlist):
        message = "Cannot find the list file {}.".format(inputlist)
        raise FileNotFoundError(message)
    if os.path.exists(outputlist) is True:
        os.remove(outputlist)
    file = open(inputlist, "r")
    tempfile = open(outputlist, "w")
    for line in file:
        line = line.rstrip()
        if (len(keyword) and len(subword)) is not 0:
            for (kword, sword) in zip(keyword, subword):
                #print(line)
                #print(kword, sword)
                line = line.replace(kword, sword)
        phy_path = rootdir + line
        tempfile.write("%s\n" % (phy_path))
    file.close()
    tempfile.close()
    return num_files
    
# TESTING LIST PROCESSING
def _templist_eval(inputlist, outputlist, outdir, overwrite=False,
                  rootdir="", keyword="", subword="", feat_format="h5"):
    num_files = 0
    if not os.path.exists(inputlist):
        message = "Cannot find the list file {}.".format(inputlist)
        raise FileNotFoundError(message)
    if os.path.exists(outputlist) is True:
        os.remove(outputlist)
    file = open(inputlist, "r")
    tempfile = open(outputlist, "w")
    for line in file:
        line = line.rstrip()
        if (len(keyword) and len(subword)) is not 0:
            for (kword, sword) in zip(keyword, subword):
                line = line.replace(kword, sword)
        phy_path = rootdir + line
        feat_id = os.path.basename(phy_path).replace(".%s" % feat_format, "")
        outputfile = outdir.replace("feat_id", feat_id)
        if os.path.exists(outputfile):
            if overwrite:
                print("over write %s" % (outputfile))
            else:
                print("%s skipped" % (outputfile))
                continue
        tempfile.write("%s\n" % (phy_path))
        num_files += 1
    file.close()
    tempfile.close()
    return num_files

# REMOVE TEMP FILE
def _remove_temp_file(file_list):
    for file in file_list:
        if os.path.exists(file):
            os.remove(file)
