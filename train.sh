wd='model/batchsize1'
nohup python src/train.py --gpu 6 --train-manifest data/pad_valid.csv --val-manifest data/record300.csv --save-folder $wd/ > $wd/out &

