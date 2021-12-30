fs=16000
stage=0
stop_stage=100

train=train
dev=dev
test=test

exp=exp
train_conf=conf/mosnet_v1.yaml

. ./utils/parse_options.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for feat in spectrogram; do
        for name in $train $dev $test; do
            python utils/extract_features.py --exp $exp \
                                             --feat $feat \
                                             --set $name \
                                             --feat_conf conf/${feat}.yaml
        done
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    python main.py --train $train --dev $dev  --conf $train_conf --exp $exp
fi

