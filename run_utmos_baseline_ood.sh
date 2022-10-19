#!/bin/bash
fs=16000
stage=1
stop_stage=100

train=ood_train
dev=ood_val
test=ood_test

features=features
feat_conf=conf/feat_extract_wav2vec.yaml
train_conf=conf/UTMOS/utmos_v1_score.yaml
extract_feature_online="False"
resume=""
debug=""
start_testing=0
end_testing=500
. ./utils/parse_options.sh

set -u
set -x
if [ "$debug" == "True" ]; then
    echo "Use debug mode"
    train=sub_train
    dev=sub_dev
    test=sub_test
    #train_conf=conf/mosnet_v1_debug.yaml
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: make list"
    bash local/make_list.sh
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: extract features"
    if [ $extract_feature_online == "True" ]; then
        echo "Since extract_feature_online set to True, skip extracting features"
    else
        for name in $train $dev; do
                python -u extract_features.py --outdir $features \
                                                --set $name \
                                                --conf $feat_conf
        done
    fi
fi

expdir=exp/${train}_`basename ${train_conf%%.*}`
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "stage 1: training mosnet"
    python -u main.py --train $train --dev $dev  --conf $train_conf --features $features --exp $expdir --device cuda --resume "$resume" --extract_feature_online $extract_feature_online
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: extract features"
    if [ $extract_feature_online == "True" ]; then
        echo "Since extract_feature_online set to True, skip extracting features"
    else
        for name in $test; do
                python extract_features.py --outdir $features \
                                                --set $name \
                                                --conf $feat_conf
        done
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "stage 3: testing mosnet"
    python test.py --test $test --conf $train_conf --exp $expdir --start $start_testing --end $end_testing --features $features --device cuda --extract_feature_online $extract_feature_online
fi
