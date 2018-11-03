# 实现中文语音识别功能，完成基于socket的在线语音识别

## 一、环境配置
### 系统环境
如果使用ubuntu系统：
```
使用ubuntu16.04，并安装配置CUDA9.0、cudnn7及python3.6.5
```
如果docker容器：
```
在https://hub.docker.com/r/nvidia/cuda/tags/上拉取镜像文件9.0-cudnn7-runtime-ubuntu16.04；
并起一个docker容器，在起docker容器时要配置-e PYTHONIOENCODING=utf-8 --net host，否则后面会报错；
然后安装python3.6.5
```
### 需安装的python3 组件:
```
python-levenshtein == 0.12.0
torch == 0.4.1
librosa == 0.6.2
ConfigParser == 3.5.0
torchaudio == 0.1      从http://github.com/pytorch/audio.git　安装
ctcdecode == 0.3      从166:/home/chuyan/ctcdecode/拷贝 运行其中的run.sh
```
### configs/test.config 配置：
```
在test.config的test字段中有三个属性：
model_path,lm_path 指示出神经网络模型，语言模型的路径；
lm_path= 表示不使用语言模型，使用greedy decoder
gpu　指示出使用的显卡号，目前仅使用一块卡，如"gpu=2"使用2号显卡,若使用cpu，则"gpu="
```

## 二、测试程序：
### 测试一个wav文件用例：
```
进入sst-api目录下，执行
./run_test.sh test.wav
```
### 测试一个数据集的cer值，输入必须为csv文件：
```
进入sst-api目录下，执行
./run_test.sh test_cer.csv
执行结果：
average cer: 0.1979112386664264
```
### 测试基于socket的在线语音识别：
```
1.运行环境为服务器：
(1)将sst-api/src/server_socket.py中的第23行sk.bind(("*.*.*.*",8080))及第59行sk.bind(("*.*.*.*",20003))中的IP地址配置为服务器IP地址；
(2)将sst-api/src/client.py放置于客户端，并将其第24行self.s.connect(('*.*.*.*', 8080))及第82行self.s.connect(('*.*.*.*', 20003))中的IP地址配置为服务器IP地址
2.运行环境为docker容器：
(1)将sst-api/src/server_socket.py中的第23行sk.bind(("*.*.*.*",8080))及第59行sk.bind(("*.*.*.*",20003))中的IP地址配置为0.0.0.0
(2)将sst-api/src/client.py放置于客户端，并将其第24行self.s.connect(('*.*.*.*', 8080))及第82行self.s.connect(('*.*.*.*', 20003))中的IP地址配置为宿主机服务器IP地址
3.进入sst-api目录下，执行
./server.sh
```

## 三、语音识别接口：
predictor.py中的Predictor类实例化语音识别，其中的predict方法提取语音识别功能，predict方法的输入为wav文件的路径，输出识别出的文本 
```
from src.predictor import Predictor
test_predictor = Predictor(conf_path)
test_predictor.predict(audio_path)
#其中conf_path为配置文件路径，conf_path='configs/test.config'
#audio_path为音频数据路径

```
## 四、模型训练：
```
./train.sh

```

