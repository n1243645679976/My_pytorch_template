dataset=/mnt/md0/user_n124345679976/dataset/BVCC/data

for name in train dev test; do
    mkdir -p data/${name}
    NAME=`echo $name | tr '[:lower:]' '[:upper:]'`
    cat $dataset/sets/${NAME}SET | cut -d',' -f2-3 | tr ',' ' ' | awk '{a[$1]+=1;print $1"_"a[$1]" "$2 }' > data/${name}/score.txt
    cat $dataset/sets/${NAME}SET | cut -d',' -f2 | tr ',\0' '  ' | awk '{a[$1]+=1;print $1"_"a[$1]" '$dataset'/wav/"$1".wav"}' > data/${name}/wav.scp
#    cp -r data/${name} data/${name}_mean
#    cat data/${name}_mean/score.txt | 
done

