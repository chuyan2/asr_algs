CUDA_VISIBLE_DEVICES=3
python transcribe.py --model-path ~/ds_bu/model/tts200000/final.pth --audio-path ~/cy_audio_data/wav/01.wav --lm-path ~/kenlm/build/big_2.arpa --cuda

#export NCCL_DEBUG=INFO
#nohup python train.py --gpu 1 --log-file ttsall/train.log  --device-ids 0 --cuda --train-manifest data/public_tts_tv_shuffle.csv --val-manifest data/mandarin_valid_manifest.csv --save-folder ttsall/ --model-path ttsall/final.pth > ttsall/ttsall_2.out &

#export NCCL_DEBUG=INFO
#python train.py --gpu 2,3,4 --cuda --train-manifest data/mandarin_valid_manifest.csv --val-manifest data/mandarin_valid_manifest.csv --save-folder tts200000_2/ --model-path tts200000_2/final.pth


#nohup python train.py --log-file tts200000_2/train.log --gpu 1 --cuda --train-manifest data/mandarin_train_manifest.csv --val-manifest data/mandarin_valid_manifest.csv --save-folder tts200000_2/ --model-path tts200000_2/final.pth > tts200000_2/train.out &
#nohup python train.py --log-file tts200000_2/train.log --gpu 1 --cuda --train-manifest data/public_tts200000_tv_shuffle.csv --val-manifest data/mandarin_valid_manifest.csv --save-folder tts200000_2/ --model-path tts200000_2/final.pth > tts200000_2/train.out &

#nohup python train.py --log-file ttsall/train.log --gpu 1,3 --cuda --train-manifest data/public_tts_tv_shuffle.csv --val-manifest data/mandarin_valid_manifest.csv --save-folder ttsall/ --model-path ttsall/final.pth > ttsall/train.out &

#python train.py --gpu 1 --device-ids 0 --cuda --train-manifest data/public_tts200000_tv_shuffle.csv --val-manifest data/mandarin_valid_manifest.csv --save-folder tts200000/ --model-path tts200000/final.pth --continue-from tts200000/final.pth

#python train.py --gpu 2,3 --cuda --train-manifest data/public_tts200000_tv_shuffle.csv --val-manifest data/mandarin_valid_manifest.csv --save-folder traintest/ --model-path traintest/final.pth
#python train.py --train-manifest data/train_manifest.csv --val-manifest data/mandarin_valid_manifest.csv --cuda
