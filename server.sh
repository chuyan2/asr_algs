export PYTHONPATH=$PWD'/src/'
rm -rf wav_tmp
mkdir wav_tmp
python src/server_socket.py configs/test.config
