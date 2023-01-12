#!/bin/bash
fs=16000
stages="1"

train=main_train
dev=main_val
test=main_test

features=features
feat_conf=conf/feat_extract_wav2vec.yaml
train_conf=conf/UTMOS/utmos_v1_score.yaml
test_conf=conf/UTMOS/utmos_v1_score_test.yaml
extract_feature_online="False"
resume=""
debug=""
tag=""
test_iter=
start_testing=
end_testing=
device="cuda"
id_dir=""
outdir=""
. ./utils/parse_options.sh

set -u
if [ "$debug" == "True" ]; then
    echo "Use debug mode"
    train=sub_train
    dev=sub_dev
    test=sub_test
    #train_conf=conf/mosnet_v1_debug.yaml
fi

if [ -z "$start_testing" ] && [ -z "$end_testing" ]; then
    start_testing=$test_iter
    end_testing=$test_iter
fi

for stage in $stages; do
if [ ${stage} -eq -1 ]; then
    echo "stage -1: make list"
    bash local/make_list.sh
fi


if [ ${stage} -eq 0 ]; then
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

if [ -z "$outdir" ]; then
    if [ -z "$tag" ]; then
        expdir=exp/${train}_`basename ${train_conf%%.*}`
    else
        expdir=exp/${train}_$tag
    fi
else
    expdir=$outdir
    if [ -d "$outdir" ]; then
        echo "WARNING: $outdir exists, things will be overwriten";
    fi
    if [ ! -d "$outdir/data_ids" ]; then
        mkdir -p $outdir/data_ids;
    fi
fi


if [ $stage -eq 1 ]; then
    echo "stage 1: training mosnet"
    python -u main.py --train $train --dev $dev  --conf $train_conf --features $features --exp $expdir --device $device --resume "$resume" --extract_feature_online $extract_feature_online
fi

if [ ${stage} -eq 2 ]; then
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

if [ -z $id_dir ]; then
    id_dir=$expdir
fi

if [ $stage -eq 3 ]; then
    echo "stage 3: testing mosnet"
    python -u test.py --id_dir $id_dir --test $test --conf $test_conf --exp $expdir --start $start_testing --end $end_testing --features $features --device $device --extract_feature_online $extract_feature_online
fi
done
