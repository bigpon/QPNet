[![Python Version](https://img.shields.io/badge/Python-3.5%2C%203.6-green.svg)](https://img.shields.io/badge/Python-3.5%2C%203.6-green.svg)

# Quasi-Periodic WaveNet (QPNet)

## Introduction
The repository is the official QPNet [[1](https://arxiv.org/abs/1907.00797), [2](https://arxiv.org/abs/2007.05663)] implementation with Pytorch.  

The generated samples can be found on our [Demo](https://bigpon.github.io/QuasiPeriodicWaveNet_demo) page.  

The repository includes two parts:
1. **Acoustic feature extraction**  
to extract spectral and prosodic features by WORLD  
2. **QPNet vocoder** (*SI: speaker-independent; SD: speaker-dependent*)  
to generate speech based on the input acoustic features

## Requirements
This repository is tested on
- Python 3.6
- Cuda 10.0
- Pytorch 1.3
- torchvision 0.4.1

## Setup
The code works with both anaconda and virtualenv.  
The following example uses anaconda.
```bash
$ conda create -n venvQPNet python=3.6
$ source activate venvQPNet
$ pip install sprocket-vc
$ pip install torch torchvision
$ git clone https://github.com/bigpon/QPNet.git
```

## Folder architecture
- **corpus**  
the folder to put corpora  
-- each corpus subfolder includes a ***scp*** subfolder for file lists and a ***wav*** subfolder for speech files 
- **qpnet_models**  
the folder for trained models
- **qpnet_output**  
the folder for decoding output files
- **src**  
the folder for source code

## Example

### Corpus download:
- Dowdlod the [Voice Conversion Challenge 2018](https://datashare.is.ed.ac.uk/handle/10283/3061) (VCC2018) corpus to run the QPNet example
```bash
$ cd QPNet/corpus/VCC2018/wav/

$ wget -o train.log -O train.zip https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_training.zip

$ wget -o eval.log -O eval.zip https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation.zip

$ wget -o ref.log -O ref.zip https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_reference.zip

$ unzip train.zip
$ unzip eval.zip
$ unzip ref.zip
```
- **SI-QPNet training set**: `corpus/VCC2018/scp/vcc18tr.scp`  
- **SD-QPNet updating set**: `corpus/VCC2018/scp/vcc18up_VCC2SPK.scp`  
- **SD-QPNet validation set**: `corpus/VCC2018/scp/vcc18va_VCC2SPK.scp`  
- **Testing  set**: `corpus/VCC2018/scp/vcc18eval.scp`  

### Path setup:
- Modify the corresponding CUDA and project root paths in `src/utils/param_path.py`
 ``` bash 
# move to the source code folder to run the following scripts
$ cd QPNet/src/
 ```  

### Feature extraction:  
1. Output the F0 and power distributions histogram figures to `corpus/VCC2018/hist/`  
 ``` bash  
$ bash run_FE.sh --stage 0
 ```  

2. Modify the **f0_min** (*lower bound of F0 range*), **f0_max** (*upper bound of F0 range*), and **pow_th** (*power threshold for VAD*) values of the speakers in `corpus/VCC2018/conf/pow_f0_dict.yml`  
*The F0 ranges setting details can be found [here](https://github.com/k2kobayashi/sprocket/blob/master/docs/vc_example.md).  

3. Extract and save acoustic features of the training, evaluation, and reference sets in `corpus/VCC2018/h5/`   
*The analysis-synthesis speech files of the training set are also saved in `corpus/VCC2018/h5_restored/`.
 ``` bash 
$ bash run_FE.sh --stage 123
 ```  

4. Process waveform files by noise shaping for QPNet training and save the shaped files in `corpus/VCC2018/wav_h5_ns/`   
 ``` bash
$ bash run_FE.sh -stage 4 
 ```

### QPNet vocoder:
1. Train and test SI-QPNet   
 ``` bash 
# the gpu ID can be set by --gpu GPU_ID (default: 0)
$ bash run_QP.sh --gpu 0 --stage 03
 ```  

2. Update SD-QPNet for each speaker with the corresponding partial training data
 ``` bash 
$ bash run_QP.sh --gpu 0 --stage 1
 ```  

3. Validate SD-QPNet for each speaker with the corresponding partial training data
 ``` bash 
# the validation results are in `qpnet_models/modelname/validation_result.yml`
$ bash run_QP.sh --gpu 0 --stage 2
 ```  

4. Test SD-QPNet with the updating iteration number according to the validation results 
 ``` bash 
# the iter number can be set by --miter NUM (default: 1000)
$ bash run_QP.sh --gpu 0 --miter 1000 --stage 4
 ```

5. Test SI-QPNet with scaled F0 (0.5 * F0 and 1.5 * F0) 
 ``` bash 
# default F0 scaled factors=("0.50" "1.50")
# the scaled factors can be changed in run_QP.sh
$ bash run_QP.sh --gpu 0 --stage 5
 ```    

6. Test SD-QPNet with scaled F0 (0.5 * F0 and 1.5 * F0) 
 ``` bash 
$ bash run_QP.sh --gpu 0 --miter 1000 --stage 6
 ```    

## Hints

- The program only support WORLD acoustic features now, but you can modify the feature extraction script and change the '**feature_type**' in `src/runFE.py` and `src/runQP.py` for new features.

- You can extract acoustic feature with different settings (ex: frame length ...) and set different '**feature_format**' (default: h5) in `src/runFE.py` and `src/runQP.py` for each setting, and the program will create the corresponding folders. 

- You can easily change the generation model by setting different '**network**' (default: qpnet) in `src/runQP.py` when you create new generation models.

- When working with new corpus, You only need to create the file lists of wav files because the program will create feature list based on the wav file list.

- When you create the wav file lists, please follow the form as the example  
(ex: rootpath/wav/xxx/xxx.wav).

## Models and results

- The pre-trained models and generated utterances are released.
- You can download all pre-trained models via the [link](https://drive.google.com/drive/folders/1HghyuYG4V0_KTBwUB1KxwZtZuzzGVgCM?usp=sharing).
- Please put the downloaded models in the `qpnet_models` folder.
- You can download all generated utterances via the [link](https://drive.google.com/drive/folders/1VcoKBPk5kjvueE7oUsmbZUdalqxoCKeM?usp=sharing).
- The released models are only trained with the vcc18 corpus (~ 1 hr). 
- To achieve higher speech qualities, more training data is required. (In our papers, the training data was ~ 3 hrs)


<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax">Corpus</th>
    <th class="tg-0lax">Language</th>
    <th class="tg-0lax">Fs [Hz]</th>
    <th class="tg-0lax">Feature</th>
    <th class="tg-0lax">Model</th>
    <th class="tg-0lax">Result</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="5">
    <a href="https://datashare.is.ed.ac.uk/handle/10283/3061">vcc18</a></td>
    <td class="tg-0pky" rowspan="5">EN</td>
    <td class="tg-0pky" rowspan="5">22050</td>
    <td class="tg-0pky" rowspan="5">world<br>(uv + f0 + mcep + ap)<br>(shiftms: 5)</td>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/11lCmaiNCoHVvtdeHW2GqKBjt6Gh5O1d0?usp=sharing">
    SI</a></td>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1O5uKoezXIvOfSHd3mTpRaSG5lvyOu6NO?usp=sharing">
    link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1-AZFn-bo3s2PxJ8O77353aYqpu-QQ0WW?usp=sharing">
    SD_VCC2SF3</td>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1Meh_upWzJz1xAig5_d5Gb1gNhSTbg0Kz?usp=sharing">
    link</td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1MuMYhPWWr7q9S7GtgsEX28WRgwrjtDVo?usp=sharing">
    SD_VCC2SF4</td>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1cqm7XhUZNCMFzF8oaPrxkkLeqbPno-25?usp=sharing">
    link</td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1qGoxHRXLDCxtpfwhf9e_tQM-1EfYp3HS?usp=sharing">
    SD_VCC2SM3</td>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1CKG1XQJ2kx9_O8cGjASFcb1j6V1JdO9j?usp=sharing">
    link</td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/11E82g6vjXysEr3rjqcyGvd9lQLAuAmyV?usp=sharing">
    SD_VCC2SM4</td>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1dQEstXnF18eLlGGfFRDt-v874yjhNkJ1?usp=sharing">
    link</td>
  </tr>
</tbody>
</table>


## References
The QPNet repository is developed based on 
- [Pytorch WaveNet](https://github.com/kan-bayashi/PytorchWaveNetVocoder) implementation by [@kan-bayashi](https://github.com/kan-bayashi)
- [Voice conversion](https://github.com/k2kobayashi/sprocket) implementation by [@k2kobayashi](https://github.com/k2kobayashi)

## Citation

If you find the code is helpful, please cite the following papers.

```
@InProceedings{qpnet_2019,
author="Y.-C. Wu and T. Hayashi and P. L. Tobing and K. Kobayashi and T. Toda",
title="Quasi-periodic {W}ave{N}et vocoder: A pitch dependent dilated convolution model for parametric speech generation",
booktitle="Proc. Interspeech",
year="2019",
month="Sept.",
pages="196-200"
}

@Article{qpnet_2020,
author="Y.-C. Wu and T. Hayashi and P. L. Tobing and K. Kobayashi and T. Toda",
title="Quasi-periodic {W}ave{N}et: An autoregressive raw waveform generative model with pitch-dependent dilated convolution neural network",
journal="IEEE/ACM Transactions on Audio, Speech, and Language Processing", 
year="(submitted)"
}
```

## Authors

Development:   
Yi-Chiao Wu @ Nagoya University ([@bigpon](https://github.com/bigpon))  
E-mail: `yichiao.wu@g.sp.m.is.nagoya-u.ac.jp`  

Advisor:  
Tomoki Toda @ Nagoya University  
E-mail: `tomoki@icts.nagoya-u.ac.jp`



