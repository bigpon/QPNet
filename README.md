[![Python Version](https://img.shields.io/badge/Python-3.5%2C%203.6-green.svg)](https://img.shields.io/badge/Python-3.5%2C%203.6-green.svg)

# Quasi-Periodic WaveNet (QPNet)

## Introduction
The repository is the [QPNet](https://arxiv.org/abs/1907.00797) implementation with Pytorch.  

The generated samples can be found at our [Demo](https://bigpon.github.io/QuasiPeriodicWaveNet_demo) page.  

The repository includes three parts:
1. **Acoustic feature extraction**  
to extract spectral and prosodic features by WORLD  
2. **QPNet vocoder** (*SI: speaker-independent; SD: speaker-dependent*)  
to generate speech based on the input acoustic features
3. **QPNet for sinusoid generation** [ongoing]  
a toy demo for generating periodic sinusoid

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

## Hints

- The program only support WORLD acoustic features now, but you can modify the feature extraction script and change the '**feature_type**' in `src/runFE.py` and `src/runQP.py` for new features.

- You can extract acoustic feature with different settings (ex: frame length ...) and set different '**feature_format**' (default: h5) in `src/runFE.py` and `src/runQP.py` for each setting, and the program will create the corresponding folders. 

- You can easily change the generation model by setting different '**network**' (default: qpnet) in `src/runQP.py` when you create new generation models.

- When working with new corpus, You only need to create the file lists of wav files because the program will create feature list based on the wav file list.

- When you create the wav file lists, please follow the form as the example  
(ex: rootpath/wav/xxx/xxx.wav).

## References

Please cite the following article.

```
@inproceedings{wu2019qpnet,
  title={Quasi-Periodic WaveNet vocoder: a pitch dependent dilated convolution model for parametric speech generation},
  author={Wu, Yi-Chiao and Hayashi, Tomoki and Patrick Lumban, Tobing and Kobayashi, Kazuhiro and Toda, Tomoki},
  booktitle={Proceedings of Interspeech},
  year={2019}
}
```

## ACKNOWLEDGEMENTS
The QPNet repository is developed based on 
- [Pytorch WaveNet](https://github.com/kan-bayashi/PytorchWaveNetVocoder) implementation by [@kan-bayashi](https://github.com/kan-bayashi)
- [Voice conversion](https://github.com/k2kobayashi/sprocket) implementation by [@k2kobayashi](https://github.com/k2kobayashi)

## Author
Yi-Chiao Wu @ Nagoya University  
E-mail: `yichiao.wu@g.sp.m.is.nagoya-u.ac.jp`



