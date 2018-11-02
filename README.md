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
ctcdecode == 0.3      从166:/home/chuyan/ctcdecode/拷贝 运行其中的run.sh
```
### configs/test.config 配置：
```
在test.config的test字段中有三个属性：
model_path,lm_path 指示出神经网络模型，语言模型的路径；
gpu　指示出使用的显卡号，目前仅使用一块卡，如"gpu=2"使用2号显卡,若使用cpu，则"gpu="
```

### 训练模型
./train.sh

### 测试模型
./run_test.sh 10c1.wav 
或
./run_test.sh x.csv
