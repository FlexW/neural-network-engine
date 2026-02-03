# Neural Network Engine

Checkout the [blog post](https://flexw.github.io/posts/building-a-neural-network-engine/).

## Building

Only tested on ArchLinux. Compile the engine by invoking `./build.sh`. For optimzed builds `./build.sh release`.

For running the Python driver make sure to install the required dependencies. It may make sense to install them in a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Example to run MobileNetV2:

```sh
./src/driver.py -config data/mobilenetv2.cfg -input data/mobilenetv2_raw_images/img_08.raw -family mobilenetv2 -classes data/imagenet_classes.txt
```

Example to run MNIST:

```sh
./driver.py -config data/mnist_simple.cfg -input data/mnist_raw_images/image_3.ubyte -family mnist
```
