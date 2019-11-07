#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Wu Yi-Chiao (Nagoya University)
# based on a WaveNet script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/PytorchWaveNetVocoder)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import logging
import sys
import time

import yaml
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from numpy.matlib import repmat

def encode_mu_law(x, mu=256):
    """PERFORM MU-LAW ENCODING
    Args:
        x (ndarray): audio signal with the range from -1 to 1
        mu (int): quantized level
    Return:
        (ndarray): quantized audio signal with the range from 0 to mu - 1
    """
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5).astype(np.int64)

def decode_mu_law(y, mu=256):
    """PERFORM MU-LAW DECODING
    Args:
        x (ndarray): quantized audio signal with the range from 0 to mu - 1
        mu (int): quantized level
    Return:
        (ndarray): audio signal with the range from -1 to 1
    """
    mu = mu - 1
    fx = (y - 0.5) / mu * 2 - 1
    x = np.sign(fx) / mu * ((1 + mu) ** np.abs(fx) - 1)
    return x

def initialize(m):
    """INITILIZE CONV WITH XAVIER
    Arg:
        m (torch.nn.Module): torch nn module instance
    """
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

class OneHot(nn.Module):
    """CONVERT TO ONE-HOT VECTOR
    Arg:
        depth (int): dimension of one-hot vector
    """
    def __init__(self, depth):
        super(OneHot, self).__init__()
        self.depth = depth

    def forward(self, x):
        """Forward calculation
        Arg:
            x (tensor): long tensor variable with the shape  (B x T)
        Return:
            (tensor): float tensor variable with the shape (B x T x depth)
        """
        x = x % self.depth
        x = torch.unsqueeze(x, 2)
        x_onehot = x.new_zeros(x.size(0), x.size(1), self.depth).float()
        return x_onehot.scatter_(2, x, 1)

def FDilatedConv1d(xC, xP, nnModule):
    """1D DILATED CAUSAL CONVOLUTION"""
    convC = nnModule.convC # current
    convP = nnModule.convP # previous
    output = F.conv1d(xC, convC.weight, convC.bias) + \
             F.conv1d(xP, convP.weight, convP.bias)
    return output

class DilatedConv1d(nn.Module):
    """1D DILATED CAUSAL CONVOLUTION"""
    def __init__(self, in_channels, out_channels, bias=True):
        super(DilatedConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convC = nn.Conv1d(in_channels, out_channels, 1, bias=bias) # 1 x 1 conv filter of current sample
        self.convP = nn.Conv1d(in_channels, out_channels, 1, bias=bias) # 1 x 1 conv filter of previous sample

    def forward(self, xC, xP):
        """Forward calculation
        Arg:
            xC (tensor): float tensor variable with the shape  (B x C x T)
            xP (tensor): float tensor variable with the shape  (B x C x T)
        Return:
            (tensor): float tensor variable with the shape (B x C x T)
        """
        xC = self.convC(xC)
        xP = self.convP(xP)
        return xC + xP 

class CausalConv1d(nn.Module):
    """1D DILATED CAUSAL CONVOLUTION"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        #self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0, dilation=dilation, bias=bias)

    def forward(self, x):
        """Forward calculation
        Arg:
            x (tensor): float tensor variable with the shape  (B x C x T)
        Return:
            (tensor): float tensor variable with the shape (B x C x T)
        """
        #x = F.pad(x, (self.padding, 0), "constant", 0)
        x = self.conv(x)
        return x

class UpSampling(nn.Module):
    """UPSAMPLING LAYER WITH DECONVOLUTION
    Arg:
        upsampling_factor (int): upsampling factor
    """
    def __init__(self, upsampling_factor, bias=True):
        super(UpSampling, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.bias = bias
        self.conv = nn.ConvTranspose2d(1, 1,
                                       kernel_size=(1, self.upsampling_factor),
                                       stride=(1, self.upsampling_factor),
                                       bias=self.bias)

    def forward(self, x):
        """Forward calculation
        Arg:
            x (tensor): float tensor variable with the shape  (B x C x T)
        Return:
            (tensor): float tensor variable with the shape (B x C x T')
                        where T' = T * upsampling_factor
        """
        x = x.unsqueeze(1)  # B x 1 x C x T
        x = self.conv(x)  # B x 1 x C x T'
        return x.squeeze(1)

class QPNet(nn.Module):
    """QUASI-PERIODIC WAVENET
    Args:
        n_quantize (int): number of quantization
        n_aux (int): number of aux feature dimension
        n_resch (int): number of filter channels for residual block
        n_skipch (int): number of filter channels for skip connection
        dilationF_depth (int): number of fixed-dilation depth (e.g. if set 10, max dilation = 2^(10-1))
        dilationF_repeat (int): number of fixed-dilation repeat
        dilationA_depth (int): number of adaptive-dilation depth (e.g. if set 10, max dilation = 2^(10-1))
        dilationA_repeat (int): number of adaptive-dilation repeat
        kernel_size (int): filter size of dilated causal convolution
        upsampling_factor (int): upsampling factor
    """
    def __init__(self, n_quantize=256, n_aux=39, 
                 n_resch=512, n_skipch=256,
                 dilationF_depth=4, dilationF_repeat=3, 
                 dilationA_depth=4, dilationA_repeat=1, 
                 kernel_size=2, upsampling_factor=110):
        super(QPNet, self).__init__()
        self.n_quantize        = n_quantize
        self.n_aux             = n_aux
        self.n_resch           = n_resch
        self.n_skipch          = n_skipch
        self.kernel_size       = kernel_size
        self.upsampling_factor = upsampling_factor
        # causal_layer
        self.receptiveCausal_field = (self.kernel_size - 1)
        # fixed dilation
        self.dilationF_depth  = dilationF_depth
        self.dilationF_repeat = dilationF_repeat
        self.dilationsF = [2 ** i for i in range(self.dilationF_depth)] 
        self.dilationsF *=  self.dilationF_repeat
        self.receptiveF_field = (self.kernel_size - 1) * sum(self.dilationsF)
        # adaptive dilation
        self.dilationA_depth  = dilationA_depth
        self.dilationA_repeat = dilationA_repeat
        self.dilationsA = [2 ** i for i in range(self.dilationA_depth)] 
        self.dilationsA *= self.dilationA_repeat
        self.receptiveA_field = (self.kernel_size - 1) * sum(self.dilationsA)
        # preprocessing
        self.onehot = OneHot(self.n_quantize)
        self.causal = CausalConv1d(self.n_quantize, self.n_resch, self.kernel_size)
        if self.upsampling_factor > 0:
            self.upsampling = UpSampling(self.upsampling_factor)
        # fixed residual blocks
        self.dilF_sigmoid = nn.ModuleList()
        self.dilF_tanh = nn.ModuleList()
        self.auxF_1x1_sigmoid = nn.ModuleList()
        self.auxF_1x1_tanh = nn.ModuleList()
        self.skipF_1x1 = nn.ModuleList()
        self.resF_1x1 = nn.ModuleList()
        for d in self.dilationsF:
            self.dilF_sigmoid += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.dilF_tanh += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.auxF_1x1_sigmoid += [nn.Conv1d(self.n_aux, self.n_resch, 1)]
            self.auxF_1x1_tanh += [nn.Conv1d(self.n_aux, self.n_resch, 1)]
            self.skipF_1x1 += [nn.Conv1d(self.n_resch, self.n_skipch, 1)]
            self.resF_1x1 += [nn.Conv1d(self.n_resch, self.n_resch, 1)]
        # adaptive residual blocks
        self.dilA_sigmoid = nn.ModuleList()
        self.dilA_tanh = nn.ModuleList()
        self.auxA_1x1_sigmoid = nn.ModuleList()
        self.auxA_1x1_tanh = nn.ModuleList()
        self.skipA_1x1 = nn.ModuleList()
        self.resA_1x1 = nn.ModuleList()
        for d in self.dilationsA:
            self.dilA_sigmoid += [DilatedConv1d(self.n_resch, self.n_resch)]
            self.dilA_tanh += [DilatedConv1d(self.n_resch, self.n_resch)]
            self.auxA_1x1_sigmoid += [nn.Conv1d(self.n_aux, self.n_resch, 1)]
            self.auxA_1x1_tanh += [nn.Conv1d(self.n_aux, self.n_resch, 1)]
            self.skipA_1x1 += [nn.Conv1d(self.n_resch, self.n_skipch, 1)]
            self.resA_1x1 += [nn.Conv1d(self.n_resch, self.n_resch, 1)]
        # postprocessing
        self.conv_post_1 = nn.Conv1d(self.n_skipch, self.n_skipch, 1)
        self.conv_post_2 = nn.Conv1d(self.n_skipch, self.n_quantize, 1)
        # channel number initialization
        self.n_ch = self.n_resch

    def forward(self, x, h, dilated_factors, blength):
        """Forward calculation
        Args:
            x (tensor): long tensor variable with the shape  (B x T)
            h (tensor): float tensor variable with the shape  (B x n_aux x T)
            dilated_factors(tensor): float tensor of the pitch-dependent dilated factors (B x T)
            blength(tensor): integer tensor of the batch_length (B)
        Return:
            (tensor): float tensor variable with the shape (B x T x n_quantize)
        """
        # index initialization
        batch_index, ch_index = self._index_initial(1, self.n_ch)

        # initialization
        assert torch.all(blength==blength[0])
        batch_length = int(blength[0]) # shift for each layer
        max_dilated_factors = int(torch.max(dilated_factors.ceil()))
        receptiveF_field = self.receptiveF_field
        receptiveA_field = self.receptiveA_field
        receptiveA_field *= max_dilated_factors

        # preprocess
        receptive_field = receptiveA_field + receptiveF_field + self.receptiveCausal_field
        output = self._preprocess(x[:, -receptive_field - batch_length:])
        if self.upsampling_factor > 0:
            h = self.upsampling(h)

        # fixed residual blocks
        receptive_field = receptiveA_field + receptiveF_field
        outputF = output
        hindex = -(receptive_field + batch_length)
        skip_connectionsF = []
        for layer, dilation in enumerate(self.dilationsF):
            shift = int(dilation)
            hindex += shift
            outputF, skip = self._fixed_residual_forward(
                outputF,
                h[:, :, hindex:],
                self.dilF_sigmoid[layer],
                self.dilF_tanh[layer],
                self.auxF_1x1_sigmoid[layer],
                self.auxF_1x1_tanh[layer],
                self.skipF_1x1[layer],
                self.resF_1x1[layer])
            skip_connectionsF.append(skip[:, :, -batch_length:])

        # adaptive residual blocks
        outputA = outputF
        hindex = -(receptiveA_field + batch_length)
        skip_connectionsA = []
        for layer, dilation in enumerate(self.dilationsA):
            shift = int(dilation*max_dilated_factors)
            hindex += shift
            dilated_index = self._dilated_index(
                dilated_factors[:, hindex:], dilation, self.n_ch)
            assert int(abs(torch.min(dilated_index))) <= outputA.size()[-1]
            past_index = (batch_index, ch_index, dilated_index) # index of previous samples
            outputA, skip = self._adaptive_residual_forward(
                outputA[:, :, shift:], # current samples
                outputA[past_index], # corresponded previous samples
                h[:, :, hindex:],
                self.dilA_sigmoid[layer], 
                self.dilA_tanh[layer],
                self.auxA_1x1_sigmoid[layer], 
                self.auxA_1x1_tanh[layer],
                self.skipA_1x1[layer], 
                self.resA_1x1[layer])
            skip_connectionsA.append(skip[:, :, -batch_length:])

        # skip-connection 
        output = sum(skip_connectionsF) + sum(skip_connectionsA)
        output = self._postprocess(output)
        
        return output
    
    def batch_fast_generate(self, x, h, 
                            n_samples_list, dilated_factors,
                            intervals=None, mode="sampling",
                            extra_memory=False):
        """Batch fast generation
        Reference [Fast Wavenet Generation Algorithm](https://arxiv.org/abs/1611.09482)
        Args:
            x (tensor): long tensor variable with the shape  (B x T)
            h (tensor): float tensor variable with the shape  (B x max(n_samples_list) + T)
            n_samples_list (list): list of number of samples to be generated (B)
            dilated_factors: float array/tensor of the pitch-dependent dilated factors (B x T)
                *dilated_factors is numpy array when extra_memory set False.
            intervals (int): log interval
            mode (str): "sampling" or "argmax"
            extra_memory(bool): processing dilated factor in tensor format or not
                * tensor mode will accelerate the decoding but consume more memory
        Return:
            (list): list of ndarray which is generated quantized wavenform
        """
        # index initialization
        batch_size = len(n_samples_list)
        batch_index, ch_index = self._index_initial(batch_size, self.n_ch, tensor=extra_memory)
        
        # get min max length
        max_n_samples = max(n_samples_list)
        min_n_samples = min(n_samples_list)
        min_idx = np.argmin(n_samples_list)

        # upsampling
        if self.upsampling_factor > 0:
            h = self.upsampling(h)

        # get receptive field length and dilated factors
        if extra_memory:
            max_dilated_factors = int(torch.max(dilated_factors.ceil()))
        else:
            max_dilated_factors = int(np.nanmax(np.ceil(dilated_factors)))
        receptiveF_field = self.receptiveF_field
        receptiveA_field = self.receptiveA_field
        receptiveA_field *= max_dilated_factors
        # padding 
        receptive_field = receptiveA_field + receptiveF_field + self.receptiveCausal_field
        n_pad = int(receptive_field - x.size(1)) + 1 # pad one more to reserve the last one for the seed
        if n_pad > 0:
            x = F.pad(x, (n_pad, 0), "constant", self.n_quantize // 2)
            h = F.pad(h, (n_pad, 0), "replicate")
            if extra_memory:
                dilated_factors = F.pad(dilated_factors, (n_pad, 0), "constant", 1.0)
            else:
                dilated_factors = np.pad(dilated_factors, ((0, 0), (n_pad, 0)), "constant", 
                                         constant_values=((1.0, 1.0), (1.0, 1.0)))
       
        # prepare causal layer buffer
        causal_output = self._preprocess(x[:, :-1]) # forward calculation except the seed
        h_ = h[:, :, :causal_output.size(-1)]
        d_ = dilated_factors[:, :causal_output.size(-1)]

        # prepare fixed network buffers
        receptive_field = receptiveA_field + receptiveF_field
        outputF = causal_output[:, :, -receptive_field:]
        hF = h_[:, :, -receptive_field:]
        hindex = -receptive_field
        outputF_buffer = []
        bufferF_size = []

        # fixed residual blocks
        for layer, dilation in enumerate(self.dilationsF):
            shift = int(dilation)
            hindex += shift
            outputF, _ = self._fixed_residual_forward(
                outputF, hF[:, :, hindex:],
                self.dilF_sigmoid[layer], self.dilF_tanh[layer],
                self.auxF_1x1_sigmoid[layer], self.auxF_1x1_tanh[layer],
                self.skipF_1x1[layer], self.resF_1x1[layer])
            if dilation == 2 ** (self.dilationF_depth - 1):
                bufferF_size.append(self.kernel_size - 1)
            else:
                bufferF_size.append(dilation * 2 * (self.kernel_size - 1))
            outputF_buffer.append(outputF[:, :, -bufferF_size[layer]:])
        outputF_A_buffer = outputF[:, :, -max_dilated_factors:]

        # prepare adaptive network buffers 
        receptive_field = receptiveA_field
        outputA = outputF[:, :, -receptive_field:]
        hA = h_[:, :, -receptive_field:]
        dA = d_[:, -receptive_field:]
        hAindex = -receptive_field
        outputA_buffer = []
        bufferA_size = []
        generate_dilated_index = []
        last_layerA = len(self.dilationsA) - 1

        # adaptive residual blocks
        for layer, dilation in enumerate(self.dilationsA):
            generate_dA_index = self._generate_dilated_index(
                dilated_factors, dilation, self.n_ch, extra_memory)
            generate_dilated_index.append(generate_dA_index)
            if layer == last_layerA:
                break
            shift = int(dilation*max_dilated_factors)
            hAindex += shift
            dA_index = self._dilated_index(dA[:, hAindex:], dilation, self.n_ch, extra_memory)
            if extra_memory:
                assert abs(torch.min(dA_index)) <= outputA.size()[-1]
            else:
                assert abs(np.amin(dA_index)) <= outputA.size()[-1]
            past_index = (batch_index, ch_index, dA_index)            
            outputA, _ = self._adaptive_residual_forward(
                outputA[:, :, shift:], # current samples
                outputA[past_index], # corresponded previous samples
                hA[:, :, hAindex:],  
                self.dilA_sigmoid[layer], 
                self.dilA_tanh[layer], 
                self.auxA_1x1_sigmoid[layer], 
                self.auxA_1x1_tanh[layer],
                self.skipA_1x1[layer], 
                self.resA_1x1[layer])
            if dilation == 2 ** (self.dilationA_depth - 1):
                bufferA_size.append(max_dilated_factors *
                                     (self.kernel_size - 1))
            else:
                bufferA_size.append(max_dilated_factors *
                                     dilation * 2 * (self.kernel_size - 1))
            outputA_buffer.append(outputA[:, :, -bufferA_size[layer]:])
            # release memory
            del dA_index
            torch.cuda.empty_cache()

        # generate
        samples = x  # B x receptive_field
        end_samples = []
        start = time.time()
        for i in range(max_n_samples):
            causal_input = samples[:, -self.kernel_size * 2 + 1:]
            causal_output = self._preprocess(causal_input) # current causal output sample
            # current auxiliary features, dilation factor
            current_index = samples.size(-1) - 1
            hC = h[:, :, current_index].contiguous().view(-1, self.n_aux, 1)
            
            # fixed network
            outputF_buffer_next = []
            skip_connectionsF = []
            outputF = causal_output[:, :, -self.kernel_size:]
            for layer, dilation in enumerate(self.dilationsF):
                outputF, skip = self._generate_fixed_residual_forward(
                    outputF, hC,
                    self.dilF_sigmoid[layer], self.dilF_tanh[layer],
                    self.auxF_1x1_sigmoid[layer], self.auxF_1x1_tanh[layer],
                    self.skipF_1x1[layer], self.resF_1x1[layer])
                skip_connectionsF.append(skip)
                outputF = torch.cat([outputF_buffer[layer], outputF], dim=2)
                outputF_buffer_next.append(outputF[:, :, -bufferF_size[layer]: ])
            # update buffer
            outputF_buffer = outputF_buffer_next

            # adaptive network
            xC = outputF[:, :, -1:]
            bufferP = outputF_A_buffer  # previous causal output samples
            outputF_A_buffer_next = \
                torch.cat([outputF_A_buffer[:, :, -max_dilated_factors + 1:], xC], dim=2)
            outputA_buffer_next = []
            skip_connectionsA = []
            for layer, dilation in enumerate(self.dilationsA):
                # previous output sample
                if extra_memory:
                    generate_dA_index = generate_dilated_index[layer][:,:,current_index].view(-1, self.n_ch, 1)
                else:
                    generate_dA_index = generate_dilated_index[layer][:,:,current_index].reshape(-1, self.n_ch, 1)
                past_index = (batch_index, ch_index, generate_dA_index)
                xP = bufferP[past_index]
                # generation residual forward
                xC, skip = self._generate_adaptive_residual_forward(
                    xC, xP, hC,
                    self.dilA_sigmoid[layer], self.dilA_tanh[layer],
                    self.auxA_1x1_sigmoid[layer], self.auxA_1x1_tanh[layer],
                    self.skipA_1x1[layer], self.resA_1x1[layer])
                skip_connectionsA.append(skip)
                if layer != last_layerA:
                    bufferP = outputA_buffer[layer]
                    outputA = torch.cat([outputA_buffer[layer], xC], dim=2)
                    outputA_buffer_next.append(outputA[:, :, -bufferA_size[layer]:])
                # release memory
                del xP
                del past_index
                del generate_dA_index
                torch.cuda.empty_cache()
            # update output buffer & F-A-buffer
            outputA_buffer = outputA_buffer_next
            outputF_A_buffer = outputF_A_buffer_next           

            # get predicted sample
            output = sum(skip_connectionsF) + sum(skip_connectionsA)
            output = self._postprocess(output)[:, -1]  # B x n_quantize
            if mode == "sampling":
                posterior = F.softmax(output, dim=-1)
                dist = torch.distributions.Categorical(posterior)
                sample = dist.sample()  # B
            elif mode == "argmax":
                sample = output.argmax(-1)  # B
            else:
                logging.error("mode should be sampling or argmax")
                sys.exit(1)
            samples = torch.cat([samples, sample.view(-1, 1)], dim=1)

            # show progress
            if intervals is not None and (i + 1) % intervals == 0:
                logging.info("%d/%d estimated time = %.3f sec (%.3f sec / sample)" % (
                    i + 1, max_n_samples,
                    (max_n_samples - i - 1) * ((time.time() - start) / intervals),
                    (time.time() - start) / intervals))
                start = time.time()

            # check length
            if (i + 1) == min_n_samples:
                while True:
                    # get finished sample
                    end_samples += [samples[min_idx, -min_n_samples:].cpu().numpy()]
                    # get index of unfinished samples
                    idx_list = [idx for idx in range(len(n_samples_list)) if idx != min_idx]
                    if len(idx_list) == 0:
                        # break when all of samples are finished
                        break
                    else:
                        # remove finished sample
                        samples = samples[idx_list]
                        h = h[idx_list]
                        outputF_A_buffer = outputF_A_buffer[idx_list]
                        dilated_factors  = dilated_factors[idx_list]
                        outputA_buffer   = [buffer_[idx_list] for buffer_ in outputA_buffer]
                        outputF_buffer   = [buffer_[idx_list] for buffer_ in outputF_buffer]
                        generate_dilated_index = [buffer_[idx_list] for buffer_ in generate_dilated_index]
                        del n_samples_list[min_idx]
                        # update index
                        batch_index = batch_index[:len(idx_list)]
                        ch_index = ch_index[:len(idx_list)]
                        # update min length
                        prev_min_n_samples = min_n_samples
                        min_n_samples = min(n_samples_list)
                        min_idx = np.argmin(n_samples_list)
                        torch.cuda.empty_cache() 
                        
                    # break when there is no same length samples
                    if min_n_samples != prev_min_n_samples:
                        break
        
        return end_samples

    def _preprocess(self, x):
        x = self.onehot(x).transpose(1, 2) # B x C x T
        output = self.causal(x)
        return output

    def _postprocess(self, x):
        output = F.relu(x)
        output = self.conv_post_1(output)
        output = F.relu(output)  # B x C x T
        output = self.conv_post_2(output).transpose(1, 2)  # B x T x C
        return output
    
    def _index_initial(self, n_batch, n_ch, tensor=True):
        batch_index = []
        for i in range(n_batch):
            batch_index.append([[i]]*n_ch)

        ch_index = []
        for i in range(n_ch):
            ch_index += [[i]]
        ch_index = [ch_index] * n_batch
        
        if tensor:
            batch_index = torch.tensor(batch_index)
            ch_index = torch.tensor(ch_index)
            if torch.cuda.is_available():
                batch_index = batch_index.cuda()
                ch_index = ch_index.cuda()

        return batch_index, ch_index
    
    def _dilated_index(self, dilated_factors, dilation, n_ch, tensor=True):
        if tensor:
            (n_batch, batch_length) = dilated_factors.size()
            sample_idx  = torch.arange(-batch_length, 0).float()
            if torch.cuda.is_available():
                sample_idx = sample_idx.cuda()
            dilations   = -dilated_factors*dilation
            dilated_idx = torch.add(dilations, sample_idx)
            dilated_idx = dilated_idx.round().long().view(n_batch, 1, batch_length)
            del sample_idx
            del dilations
            torch.cuda.empty_cache()
            return dilated_idx.repeat((1, n_ch, 1))
        else:
            (n_batch, batch_length) = dilated_factors.shape
            dilations   = -dilated_factors*dilation
            dilated_idx = np.add(dilations, np.array(range(-batch_length, 0)))
            dilated_idx = np.int32(np.round(dilated_idx))
            dilated_idx = np.matlib.repmat(dilated_idx, 1, n_ch)
            return dilated_idx.reshape(n_batch, n_ch, batch_length)

    def _generate_dilated_index(self, dilated_factor, dilation, n_ch, tensor=True):
        if tensor:
            (n_batch, batch_length) = dilated_factor.size()
            dilated_idx = -dilated_factor*dilation
            dilated_idx = dilated_idx.round().long().view(n_batch, 1, batch_length)
            return  dilated_idx.repeat((1, n_ch, 1))
        else:
            (n_batch, batch_length) = dilated_factor.shape
            dilated_idx = -dilated_factor*dilation
            dilated_idx = np.int32(np.round(dilated_idx))
            dilated_idx = repmat(dilated_idx, 1, n_ch)
            return dilated_idx.reshape(n_batch, n_ch, batch_length)

    def _adaptive_residual_forward(self, xC, xP, h,
                                  dil_sigmoid, dil_tanh,
                                  aux_1x1_sigmoid, aux_1x1_tanh, 
                                  skip_1x1, res_1x1):
        output_sigmoid = dil_sigmoid(xC, xP)
        output_tanh = dil_tanh(xC, xP)
        aux_output_sigmoid = aux_1x1_sigmoid(h)
        aux_output_tanh = aux_1x1_tanh(h)
        output = torch.sigmoid(output_sigmoid + aux_output_sigmoid) * \
            torch.tanh(output_tanh + aux_output_tanh)
        torch.tanh(output_tanh + aux_output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + xC
        return output, skip
    
    def _generate_adaptive_residual_forward(self, xC, xP, hC,
                                           dil_sigmoid, dil_tanh,
                                           aux_1x1_sigmoid, aux_1x1_tanh,
                                           skip_1x1, res_1x1):
        output_sigmoid = FDilatedConv1d(xC, xP, dil_sigmoid)
        output_tanh = FDilatedConv1d(xC, xP, dil_tanh)
        aux_output_sigmoid = aux_1x1_sigmoid(hC)
        aux_output_tanh = aux_1x1_tanh(hC)
        output = torch.sigmoid(output_sigmoid + aux_output_sigmoid) * \
            torch.tanh(output_tanh + aux_output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + xC
        return output, skip

    def _fixed_residual_forward(self, x, h, 
                          dil_sigmoid, dil_tanh,
                          aux_1x1_sigmoid, aux_1x1_tanh, 
                          skip_1x1, res_1x1):
        output_sigmoid = dil_sigmoid(x)
        output_tanh = dil_tanh(x)
        aux_output_sigmoid = aux_1x1_sigmoid(h)
        aux_output_tanh = aux_1x1_tanh(h)
        output = torch.sigmoid(output_sigmoid + aux_output_sigmoid) * \
            torch.tanh(output_tanh + aux_output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + x[:, :, -output.size(-1):]
        return output, skip

    def _generate_fixed_residual_forward(self, x, h, 
                                   dil_sigmoid, dil_tanh,
                                   aux_1x1_sigmoid, aux_1x1_tanh, 
                                   skip_1x1, res_1x1):
        output_sigmoid = dil_sigmoid(x)[:, :, -1:]
        output_tanh = dil_tanh(x)[:, :, -1:]
        aux_output_sigmoid = aux_1x1_sigmoid(h)
        aux_output_tanh = aux_1x1_tanh(h)
        output = torch.sigmoid(output_sigmoid + aux_output_sigmoid) * \
            torch.tanh(output_tanh + aux_output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + x[:, :, -1:]  # B x C x 1
        return output, skip
    
