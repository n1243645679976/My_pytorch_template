tardir=$1
set -x
if [ -d $tardir ]; then
    echo "outputdir: $tardir exist, please remove it by yourself"
    exit 0
fi
mkdir -p $tardir
shift
for sourcedir in $@; do
    for file in `ls $sourcedir | grep -v 'id$'`; do
        cat $sourcedir/$file >> $tardir/$file
    done
done
