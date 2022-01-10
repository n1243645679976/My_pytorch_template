set=$1
dB=$2
duration=$3

[ -z $dB ] && dB=-40
[ -z $duration ] && duration=0.1

echo "Extracting feature from set $set"
cat data/$set/wav.scp | while read line; do
    key=`echo ${line} | cut -d' ' -f1`
    wav=`echo ${line} | cut -d' ' -f2`
    silences=`ffmpeg -i $wav -af silencedetect=noise=${dB}dB:d=$duration -f null - 2>&1 | grep 'silence' | awk '{print $NF}' | tr '\n' ' '`
done > data/$set/silence.list
