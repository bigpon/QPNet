#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Wu Yi-Chiao (Nagoya University)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# NETWORK & TRAINING PARAMETERS INITIALIZATION
class network_parameter(object):
    def __init__(self, 
                 upsampling_flag=True, spk_code_flag=False,
                 quantize=256, aux=39, resch=512, skipch=256,
                 learningrate=1e-4, weight_decay=0.0, 
                 iters=200000, update_iters=50000,
                 checkpoint_interval=10000,
                 update_interval=5000):
        self.upsampling_flag         = upsampling_flag
        self.spk_code_flag           = spk_code_flag
        self.quantize                = quantize
        self.aux                     = aux
        self.resch                   = resch
        self.skipch                  = skipch
        self.lr                      = learningrate
        self.weight_decay            = weight_decay
        self.iters                   = iters
        self.update_iters            = update_iters
        self.checkpoint_interval     = checkpoint_interval
        self.update_interval         = update_interval

    def set_batch_param(self, batch_length=20000, batch_size=1):
        self.batch_length      = batch_length
        self.batch_size        = batch_size
    
    def set_network_ch(self, quantize=256, aux=39, resch=512, skipch=256):
        self.quantize        = quantize
        self.aux             = aux
        self.resch           = resch
        self.skipch          = skipch

class qpwn_parameter(network_parameter):
    def __init__(self, network, 
                 upsampling_flag=True, spk_code_flag=False,
                 quantize=256, aux=39, resch=512, skipch=256,
                 learningrate=1e-4, weight_decay=0.0, 
                 iters=200000, update_iters=3000,
                 checkpoint_interval=10000, 
                 update_interval=100,
                 decode_batch_size=12):
        super().__init__(upsampling_flag, spk_code_flag,
                         quantize, aux, resch, skipch, 
                         learningrate, weight_decay, 
                         iters, update_iters, 
                         checkpoint_interval,
                         update_interval)
        self._update_network(network, decode_batch_size)

    def _update_network(self, network, decode_batch_size):
        self.network = network
        if network == 'default':
            self._update_network_param(
                dilationF_depth=4, dilationF_repeat=3,
                dilationA_depth=4, dilationA_repeat=1,
                kernel_size=2, max_length=30000,
                batch_length=20000, batch_size=1,
                f0_threshold=0, decode_batch_size=decode_batch_size)
        elif network == 'Rd10Rr3Ed4Er1':
            self._update_network_param(
                dilationF_depth=10, dilationF_repeat=3,
                dilationA_depth=4, dilationA_repeat=1,
                kernel_size=2, max_length=22500,
                batch_length=20000, batch_size=1,
                f0_threshold=0, decode_batch_size=7)
        else:
            raise ValueError("%s is not supported!" % network)

    def _update_network_param(self,
                              dilationF_depth, dilationF_repeat,
                              dilationA_depth, dilationA_repeat,
                              kernel_size, max_length,
                              batch_length, batch_size,
                              f0_threshold, decode_batch_size):
        self.dilationF_depth  = dilationF_depth
        self.dilationF_repeat = dilationF_repeat
        self.dilationA_depth  = dilationA_depth
        self.dilationA_repeat = dilationA_repeat
        self.kernel_size       = kernel_size
        self.max_length        = max_length
        self.batch_length      = batch_length
        self.batch_size        = batch_size
        self.f0_threshold      = f0_threshold
        self.decode_batch_size = decode_batch_size

