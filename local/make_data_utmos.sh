
for tup in train_main, test_main,-r; do
    set=`echo $tup | cut -d',' -f1`
    com=`echo $tup | cut -d',' -f2`
    mkdir -p data/$set
    paste <(cat TRAINSET | cut -d',' -f2 | tr '/' '-') <(cat TRAINSET| cut -d',' -f2 | awk '{print "'`realpath /nas01/homes/cheng22-1000061/MOSNet/dataset/`'/" $1}') | tr '\t' ' ' | rand -n 90000 -seed 1006 $com > data/$set/wav.scp

    paste <(cat TRAINSET | cut -d',' -f2 | tr '/' '-') <(cat TRAINSET| cut -d',' -f3) | tr '\t' ' ' | rand -n 90000 -seed 1006 $com > data/$set/score.txt

    paste <(cat TRAINSET | cut -d',' -f2 | tr '/' '-') <(cat TRAINSET| cut -d',' -f5) | tr '\t' ' ' | rand -n 90000 -seed 1006 $com > data/$set/judge_id.emb

    paste <(cat TRAINSET | cut -d',' -f2 | tr '/' '-') <(cat TRAINSET| awk '{print "timit"}') | tr '\t' ' ' | rand -n 90000 -seed 1006 $com > data/$set/domain.emb

    paste <(cat concat_add_timit.txt | cut -d',' -f1 | tr '/' '-') <(cat concat_add_timit.txt | cut -d',' -f6) | tr '\t' ' ' | rand -n 90000 -seed 1006 $com > data/$set/text.listemb

    paste <(cat concat_add_timit.txt | cut -d',' -f1 | tr '/' '-') <(cat concat_add_timit.txt |  cut -d',' -f9) | tr '\t' ' ' | rand -n 90000 -seed 1006 $com > data/$set/ref.listemb
done

