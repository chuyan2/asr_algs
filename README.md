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
### 训练模型
./train.sh
训练不面向实时解码的模型，将train.py中的import model_realtime替换为import model
