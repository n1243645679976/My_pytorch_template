set -u
file1=`realpath $1`
file2=`realpath $2`

mkdir -p data/template
echo "key1 key1 key2" > data/template/trial
echo "key1 $file1" > data/template/wav.scp
echo "key2 $file2" > data/template/wav1.scp
echo "key1 aa bb c" > 
