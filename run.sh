fs=16000
stage=0
stop_stage=100

train=train_mosnet_mean
dev=dev_mosnet_mean
test=test_mosnet_mean

features=features
feat_conf=conf/feat_extract_v2.yaml
train_conf=conf/mosnet_v1.yaml
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
    train_conf=conf/mosnet_v1_debug.yaml
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
        for name in $train ; do #$dev $test; do
    #        if [ $feat == 'silence' ]; then
    #            bash utils/get_silence.sh $name
    #        else
                python utils/extract_features.py --outdir $features \
                                                --set $name \
                                                --conf $feat_conf
    #       fi
        done
    fi
fi

expdir=exp/${train}_`basename ${train_conf%%.*}`
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "stage 1: training mosnet"
    python main.py --train $train --dev $dev  --conf $train_conf --features $features --exp $expdir --device cuda --resume "$resume" --extract_feature_online $extract_feature_online
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "stage 2: testing mosnet"
    python test.py --test $test --conf $train_conf --exp $expdir --start $start_testing --end $end_testing --features $features --device cuda --extract_feature_online $extract_feature_online
fi
