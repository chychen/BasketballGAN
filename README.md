<img align="right" src="https://drive.google.com/uc?export=view&id=1W4h1WA4Lp1c_BrTmPBPwvF1Udf7SdzLO" width="200" title="A Generated Play"/>

# BasketballGAN

### Generate the ghosting defensive strategies given offensive sketch.

![](https://drive.google.com/uc?export=view&id=1lmxvBG-PTLg4vhEF_hmG1IS20vDEyvyv)

<img align="right" src="https://drive.google.com/uc?export=view&id=1QWN9BtFgaAKA1tvx_ePQku934CeCWIRl" width="500" title="A Generated Play"/><br>

## [Paper](https://arxiv.org/abs/1909.07088) | [CGVLab](https://people.cs.nctu.edu.tw/~yushuen/)<br>[Video](https://youtu.be/NTir0-znPyw) | [Supplemental](https://drive.google.com/a/nvidia.com/file/d/1dXMA_1AjpPu7J4_Iw1yb6pp-9d9Lp2uN/view?usp=sharing)

### BasketballGAN: Generating Basketball Play Simulation through Sketching

Hsin-Ying Hsieh<sup>1</sup>, Chieh-Yu Chen<sup>2</sup>, Yu-Shuen Wang<sup>1</sup> and Jung-Hong Chuang<sup>1</sup>

<sup>1</sup>National Chiao Tung University, 

<sup>2</sup>NVIDIA Corporation

Accepted paper in [ACMMM 2019](https://www.acmmm.org/2019/).

## Prerequisites

- OS: Linux
- [NVIDIA Dokcer](https://github.com/NVIDIA/nvidia-docker)
- [NVIDIA NGC Tensorflow Docker Image](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
- NVIDIA GPU (V100 16GB)

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
- save [dataset](https://drive.google.com/file/d/12DRJBIyN20vwRyfywvCXo-nNDf0UBPjZ/view?usp=sharing) under folder 'data'

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

### Public Relations

- [AAAS Science News](https://www.sciencemag.org/news/2019/09/watch-ai-help-basketball-coaches-outmaneuver-opposing-team)
- [Deeplearning.ai FB](https://www.facebook.com/deeplearningHQ/posts/1431901466962064)
- [Deeplearning.ai The Batch](https://info.deeplearning.ai/the-batch-google-achieves-quantum-supremacy-amazon-aims-to-sway-lawmakers-ai-predicts-basketball-plays-face-detector-preserves-privacy)

## Citation
If you find this useful for your research, please use the following.

``` 
@article{hsieh2019basketballgan,
  title={BasketballGAN: Generating Basketball Play Simulation Through Sketching},
  author={Hsieh, Hsin-Ying and Chen, Chieh-Yu and Wang, Yu-Shuen and Chuang, Jung-Hong},
  journal={arXiv preprint arXiv:1909.07088},
  year={2019}
}
```
