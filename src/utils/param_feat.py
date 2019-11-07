#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Wu Yi-Chiao (Nagoya University)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import math

# ACOUSTIC PARAMETER
class acoustic_parameter(object):
    def __init__(self, fs, feature_type="world",
                 shiftms=5, fftl=1024, mag=0.5,
                 mcep_dim_start=2, f0_dim_idx=1,
                 highpass_cutoff = 70, 
                 minf0=40, maxf0=800):
        self.feature_type    = feature_type
        self.shiftms         = shiftms
        self.fftl            = fftl
        self.mag             = mag
        self.mcep_dim_start  = mcep_dim_start
        self.f0_dim_idx      = f0_dim_idx
        self.highpass_cutoff = highpass_cutoff
        self._update_fs(fs)
        self._update_f0range(minf0, maxf0)
    
    def _update_f0range(self, minf0, maxf0):
        self.minf0 = minf0
        self.maxf0 = maxf0
    
    def _update_fs(self, fs):
        self.fs = str(fs)
        if self.fs == "16000":
            #mcep_dim_end = 27 #2+24+1
            #aux_dim      = 28 #(24+1)mcep + 1uv + 1f0 + 1ap
            self._update_mcep_param(mcep_alpha=0.410, aux_dim=28,
                                    mcep_dim=24, mcep_dim_end=27,
                                    ap_dim_idx=-1)
        elif self.fs == "22050":
            #mcep_dim_end = 37 #2+34+1
            #aux_dim      = 39 #(34+1)mcep + 1uv + 1f0 + 2ap
            self._update_mcep_param(mcep_alpha=0.455, aux_dim=39,
                                    mcep_dim=34, mcep_dim_end=37,
                                    ap_dim_idx=-2)
        elif self.fs == "24000":
            #mcep_dim_end = 42 #2+39+1
            #aux_dim      = 45 #(39+1)mcep + 1uv + 1f0 + 3ap
            self._update_mcep_param(mcep_alpha=0.466, aux_dim=45,
                                    mcep_dim=39, mcep_dim_end=42,
                                    ap_dim_idx=-3)
        else:
            raise ValueError("%s is not supported!" % self.fs)
        self._get_upsampling_factor(fs)

    def _update_mcep_param(self, 
                           mcep_alpha, aux_dim, 
                           mcep_dim, mcep_dim_end,
                           ap_dim_idx):
        self.mcep_alpha   = mcep_alpha
        self.aux_dim      = aux_dim
        self.mcep_dim     = mcep_dim
        self.mcep_dim_end = mcep_dim_end
        self.ap_dim_idx   = ap_dim_idx

    def _get_upsampling_factor(self, fs):
        self.upsampling_factor = math.floor(self.shiftms * float(fs) / 1000)