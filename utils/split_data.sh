data=$1
lines=`cat $data/wav.scp | wc -l`
test_lines=$(($lines / 10))
train_lines=$(($lines - $test_lines))
echo "$lines $test_lines $train_lines"
mkdir ${data%/}_test
mkdir ${data%/}_train
for f in `ls $data`; do
    cat $data/$f | rand -n $test_lines -seed 1006 | sort > ${data%/}_test/$f
    cat $data/$f | rand -n $test_lines -seed 1006 -r | sort > ${data%/}_train/$f
done