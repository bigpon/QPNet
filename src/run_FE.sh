#! /bin/bash

stage=
# stage 0: feature distribution extraction  
# stage 1: feature extraction of training set 
# stage 2: feature extraction of evaluation set 
# stage 3: feature extraction of reference set 
# stage 4: noise shaping of training waveforms
. parse_options.sh || exit 1;
hubspks=("VCC2SF1" "VCC2SF2" "VCC2SM1" "VCC2SM2")
spospks=("VCC2SF3" "VCC2SF4" "VCC2SM3" "VCC2SM4")
srcspks=("${hubspks[@]}" "${spospks[@]}")
tarspks=("VCC2TM1" "VCC2TM2" "VCC2TF1" "VCC2TF2")
allspks=("${srcspks[@]}" "${tarspks[@]}")

# Feature distribution extraction
if echo ${stage} | grep -q 0; then
    for spk in ${allspks[*]};
    do
        python runFE.py -f 22050 -e vcc18tr_$spk.scp -1 $spk
    done
fi

# Feature extraction: training set
if echo ${stage} | grep -q 1; then
    for spk in ${allspks[*]};
    do
        # feature extraction
        python runFE.py -r -i -f 22050 -e vcc18tr_$spk.scp -2 $spk
        # feature restored
        python runFE.py -r -f 22050 -e vcc18tr_$spk.scp -2 $spk
    done
fi

# Feature extraction: evaluation set
if echo ${stage} | grep -q 2; then
    for spk in ${srcspks[*]};
    do
        # feature extraction
        python runFE.py -r -i -f 22050 -e vcc18eval_$spk.scp -2 $spk
        # feature restored
        #python runFE.py -f 22050 -e vcc18eval_$spk.scp -2 $spk
    done
fi

# Feature extraction: reference set
if echo ${stage} | grep -q 3; then
    for spk in ${tarspks[*]};
    do
        # feature extraction
        python runFE.py -r -i -f 22050 -e vcc18ref_$spk.scp -2 $spk
        # feature restored
        #python runFE.py -f 22050 -e vcc18ref_$spk.scp -2 $spk
    done
fi

# Waveform noise shaping
if echo ${stage} | grep -q 4; then
    # feature statistic
    python runFE.py -r -f 22050 -e vcc18tr.scp -3 allspk    
    # noise shaping
    python runFE.py -r -f 22050 -e vcc18tr.scp -4 allspk
fi