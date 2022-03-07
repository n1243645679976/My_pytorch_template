python utils/logger.py --conf conf/mosnet_v1.yaml --train test_train --dev test_train --exp test_exp
python utils/optimizer.py --conf conf/mosnet_v1.yaml --train test_train --dev test_train --exp test_exp

python main.py --train all_hifigan --dev all_hifigan --conf conf/svsnet_v2.yaml --features features --exp exp/all_hifigan --device cuda --extract_feature_online True
python dataset.py --train all_hifigan --dev all_hifigan --conf conf/svsnet_v1.yaml --exp features --device cuda --extract_feature_online True
