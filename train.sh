wd='model/deploy/'
if [ ! -d $wd ];then
    mkdir $wd
fi
nohup python src/train.py --gpu 0 --train-manifest /home/chuyan/asr_algs/data/public_movie_shuffle.csv --val-manifest /home/chuyan/asr_algs/data/record300.csv --save-folder $wd/ > $wd/out &
tail -f $wd/train.log

