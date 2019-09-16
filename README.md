# BasketballGAN

### Generate the ghosting defensive strategies given offensive sketch.

![](https://drive.google.com/uc?export=view&id=1lmxvBG-PTLg4vhEF_hmG1IS20vDEyvyv)

<img align="right" src="https://drive.google.com/uc?export=view&id=1QWN9BtFgaAKA1tvx_ePQku934CeCWIRl" width="500" title="A Generated Play"/>

## [Paper](TODO) | [CGVLab](https://people.cs.nctu.edu.tw/~yushuen/)<br>[Video](https://drive.google.com/uc?export=view&id=1Ead1EyHdPIFFsDtanQ91w8ha-SPRkM6E) | [Supplemental](https://drive.google.com/a/nvidia.com/file/d/1dXMA_1AjpPu7J4_Iw1yb6pp-9d9Lp2uN/view?usp=sharing)

### BasketballGAN: Generating Basketball Play Simulation through Sketching

Hsin-Ying Hsieh<sup>1</sup>, Chieh-Yu Chen<sup>2</sup>, Yu-Shuen Wang<sup>1</sup> and Jung-Hong Chuang<sup>1</sup>

<sup>1</sup>National Chiao Tung University, <sup>2</sup>NVIDIA Corporation

In [ACMMM 2019](https://www.acmmm.org/2019/).

## Prerequisites

- OS: Linux
- [NVIDIA Dokcer](https://github.com/NVIDIA/nvidia-docker)
- [NVIDIA NGC Tensorflow Docker Image](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
- NVIDIA GPU (V100 16GB)

### Dataset

## Getting Stated

```bash
~$ git clone https://github.com/chychen/BasketballGAN.git
~$ cd BasketballGAN
BasketballGAN$ docker login nvcr.io
BasketballGAN$ docker pull nvcr.io/nvidia/tensorflow:19.06-py2
BasketballGAN$ docker run --runtime=nvidia -it --rm -v $PWD:$PWD --net host nvcr.io/nvidia/tensorflow:19.06-py2 bash
root@c63207c81408:~/BasketballGAN$ apt update
root@c63207c81408:~/BasketballGAN$ apt install ffmpeg
```

### Download Dataset 

- create 'data' folder
- save [dataset](https://drive.google.com/a/nvidia.com/file/d/1955WfjX2xtHVb6QAJ70zLQH65V0JD_e3/view?usp=sharing) under folder 'data'

```bash
BasketballGAN$ mkdir data
```

### Training

```bash
BasketballGAN$ cd src
BasketballGAN/src$ python Train_Triple.py --folder_path='tmp' --data_path='data'
```

### Logs/Samples/Checkpoints

```bash
- "BasketballGAN/src/tmp/Log": training summary for tensorboard.
- "BasketballGAN/src/tmp/Samples": generated videos sampled on different epoches.
- "BasketballGAN/src/tmp/Checkpoints": tensorflow checkpoints on different iterations.
```

### Monitoring

- Sampled Videos
    - Using Simple HTTP Server to monitor sampled videos while training.
    - [Simple HTTP Server (http://127.0.0.1:8000)](http://127.0.0.1:8000/tmp/Log/Samples)

```bash
BasketballGAN/src$ python -m http.server 8000
```

- Training Logs
    - [Tensorboard (127.0.0.1:6006)](http://127.0.0.1:6006)

```bash
BasketballGAN/src$ tensorboard --logdir='tmp/Log'
```

<img src="https://drive.google.com/uc?export=view&id=10NNSibWbU0oMr9ziaQeOcgft44NwBVf2" width="600" title="Earth Moving Distance"/>

## Citation
If you find this useful for your research, please use the following.

``` 
@inproceedings{hsieh2019basketballgan,
    title={BasketballGAN: Generating Basketball Play Simulation through Sketching},
    author={Hsin-Ying Hsieh, Chieh-Yu Chen, Yu-Shuen Wang and Jung-Hong Chuang},  
    booktitle={2019 ACM Multimedia Conference on Multimedia Conference},
    year={2019}
}
```
