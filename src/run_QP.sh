#! /bin/bash
gpu=0
miter=1000
stage=
# stage 0: training SI-QPNet
# stage 1: updating SD-QPNet
# stage 2: validation of SD-QPNet
# stage 3: testing w/ SI-QPNet
# stage 4: testing w/ SD-QPNet
. parse_options.sh || exit 1;
hubspks=("VCC2SF1" "VCC2SF2" "VCC2SM1" "VCC2SM2")
spospks=("VCC2SF3" "VCC2SF4" "VCC2SM3" "VCC2SM4")
srcspks=("${hubspks[@]}" "${spospks[@]}")
tarspks=("VCC2TM1" "VCC2TM2" "VCC2TF1" "VCC2TF2")
allspks=("${srcspks[@]}" "${tarspks[@]}")

# Speaker independent QPNet (SI-QPNet) training
if echo ${stage} | grep -q 0; then
    echo "SI-QPNet training."
    python runQP.py -g ${gpu} -f 22050 \
    -w vcc18tr.scp -a vcc18tr.scp -d 8 -1
fi

# Speaker dependent QPNet (SD-QPNet) updating
if echo ${stage} | grep -q 1; then
    for spk in ${spospks[*]};
    do
        echo "${spk} SD-QPNet updating."
        python runQP.py -g ${gpu} -f 22050 \
        -w vcc18tr.scp -a vcc18tr.scp \
        -x vcc18up_${spk}.scp -u vcc18up_${spk}.scp -d 8 -2
    done
fi

# Speaker dependent QPNet (SD-QPNet) validation
if echo ${stage} | grep -q 2; then
    for spk in ${spospks[*]};
    do
        echo "${spk} SD-QPNet updating."
        python runQP.py -g ${gpu} -f 22050 \
        -w vcc18tr.scp -a vcc18tr.scp \
        -x vcc18up_${spk}.scp -u vcc18up_${spk}.scp \
        -y vcc18va_${spk}.scp -v vcc18va_${spk}.scp -d 8 -5
    done
fi

# Decoding with SI-QPNet
if echo ${stage} | grep -q 3; then
    for spk in ${spospks[*]};
    do
        echo "${spk} SI-QPNet decoding."
        python runQP.py -m -g ${gpu} -f 22050 \
        -w vcc18tr.scp -a vcc18tr.scp \
        -e vcc18eval_${spk}.scp -d 8 -3 -4 ${spk}
    done
fi

# Decoding with SD-QPNet
if echo ${stage} | grep -q 4; then
    for spk in ${spospks[*]};
    do
        echo "${spk} SD-QPNet decoding."
        python runQP.py -g ${gpu} -f 22050 -M ${miter}\
        -w vcc18tr.scp -a vcc18tr.scp \
        -x vcc18up_${spk}.scp -u vcc18up_${spk}.scp \
        -e vcc18eval_${spk}.scp -d 8 -3 -4 ${spk}
    done
fi