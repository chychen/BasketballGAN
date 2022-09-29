# UI

## docker image (tested in ubuntu)

``` bash
BasketballGAN$ cd UI
~$ xhost +local:root; \
docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):$(pwd) -w $(pwd) -e DISPLAY=unix$DISPLAY jaycase/bballgan_ui:latest bash
```

## Setup

### Dataset

- [link](https://drive.google.com/drive/folders/1pa_ZDgoMWWEV8smQZrA1j9nSN7XJB8mI?usp=sharing)
- unzip then put under the path 'BasketballGAN/UI/Data/Model_data/'

### checkpoints

- trained model
	- [file1](https://drive.google.com/file/d/12DRJBIyN20vwRyfywvCXo-nNDf0UBPjZ/view?usp=sharing), [file2](https://drive.google.com/file/d/12hKIEus58BIGIsw36KjClNSS8SmJRBHc/view?usp=sharing), [file3](https://drive.google.com/file/d/1-8wmV2XMU2yPwfnijLQlvCJsdAT2j128/view?usp=sharing)
- unzip then put uner the path 'BasketballGAN/UI/Data/checkpoints/'

## start UI program

```bash
python3 Main.py
```

## UI user guide

1. place all players by drag-n-drop
2. place ball by double clicks onto nestest player
3. hit button "Set"
4. design your offensive strategies by drag-n-drop players, 
5. pass/shoot ball by drag-n-drop(click nearby ball not at the ball).
6. hit button "Generate", you will see outputs on the right hand side.
7. select buttons "Sketch Animation" and "Play Simulation" to switch mode.(and be sure "Play" button for play the animation)
