xhost +local:root; \
docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):$(pwd) -w $(pwd) -e DISPLAY=unix$DISPLAY jaycase/bballgan_ui:latest bash

