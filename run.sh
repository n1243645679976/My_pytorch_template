fs=16000
stage=0
stop_stage=100

train=train
dev=dev
test=test

features=features
train_conf=conf/mosnet_v1.yaml
resume=""
debug=""
start_testing=0
end_testing=500
. ./utils/parse_options.sh


if [ "$debug" == "True" ]; then
    train=sub_train
    dev=sub_dev
    test=sub_test
    train_conf=conf/mosnet_v1_debug.yaml
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    bash local/make_list.sh
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for feat in spectrogram ; do #silence; do #spectrogram; do
        for name in $train $dev $test; do
            if [ $feat == 'silence' ]; then
                bash utils/get_silence.sh $name
            else
                python utils/extract_features.py --outdir $features \
                                                 --feat $feat \
                                                 --set $name \
                                                 --feat_conf conf/${feat}.yaml
            fi
        done
    done
fi

expdir=exp/${train}_`basename ${train_conf%%.*}`
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    python main.py --train $train --dev $dev  --conf $train_conf --features $features --exp $expdir --device cuda --resume "$resume"
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    python test.py --test $test --conf $train_conf --exp $expdir --start $start_testing --end $end_testing --features $features --device cuda 
fi