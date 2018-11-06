wd='model/deploy/context10'
if [ ! -d $wd ];then
    mkdir $wd
fi
nohup python src/train.py --gpu 6,7 --train-manifest data/pad_valid.csv --val-manifest data/record300.csv --save-folder $wd/ > $wd/out &
tail -f $wd/train.log

