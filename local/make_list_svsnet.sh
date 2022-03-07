train_list='/mnt/md1/user_n124345679976/feature/sim_net/sim_list_train.txt'
test_list='/mnt/md1/user_n124345679976/feature/sim_net/sim_list_test.txt'
wav_path='/mnt/md1/user_n124345679976/dataset/sim_net/wav'
mkdir -p data/train_svsnet_not_mean
mkdir -p data/test_svsnet_not_mean
cat $train_list | awk -F',' '{print $1"&"$2" '$wav_path'/"$1}' | solve_dup > data/train_svsnet_not_mean/wav.scp
cat $train_list | awk -F',' '{print $1"&"$2" '$wav_path'/"$2}' | solve_dup > data/train_svsnet_not_mean/wav1.scp
cat $train_list | awk -F',' '{print $1"&"$2" "$4}' | solve_dup > data/train_svsnet_not_mean/score.txt

cat $test_list | awk -F',' '{print $1"&"$2" '$wav_path'/"$1}' | solve_dup > data/test_svsnet_not_mean/wav.scp
cat $test_list | awk -F',' '{print $1"&"$2" '$wav_path'/"$2}' | solve_dup > data/test_svsnet_not_mean/wav1.scp
cat $test_list | awk -F',' '{print $1"&"$2"  "$4}' | solve_dup > data/test_svsnet_not_mean/score.txt


