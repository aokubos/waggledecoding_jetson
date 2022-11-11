# Real-time waggle decoding system of honeybees on NVIDIA Jetson

# Environment

- NVIDIA Jetson Nano - Jetpack 4.6 [L4T 32.61]

- UVC working with 120 fps

# Install Python packages

export OPENBLAS_CORETYPE=ARMV8

sudo apt-get install python3-pip

python3 -m pip install --upgrade pip

python3 -m pip install --upgrade setuptools

python3 -m pip install --upgrade wheel

python3 -m pip install --upgrade numpy

python3 -m pip install --upgrade scipy

python3 -m pip install cython

python3 -m pip install cupy

python3 -m pip install tqdm

python3 -m pip install astropy

python3 -m pip install folium

sudo apt-get install gdal-bin

sudo apt-get install python3-gdal

## Install UVC controllor

sudo apt-get install v4l-utils

