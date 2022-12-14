# Real-time waggle decoding on NVIDIA Jetson

## Hardwares

- NVIDIA Jetson Nano - Jetpack 4.6 [L4T 32.61]

- UVC camera working with 120 fps

## Install necessary packages

### Set environment variables
```
export OPENBLAS_CORETYPE=ARMV8
export PATH=~/.local/bin:$PATH
```
### Install pip and upgrage Python packages
```
sudo apt-get install python3-pip
python3 -m pip install --upgrade pip
sudo apt-get install python3-testresources
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade protobuf
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade scipy
```
### Install Python packages
```
python3 -m pip install cython
python3 -m pip install cupy
python3 -m pip install tqdm
python3 -m pip install astropy
python3 -m pip install folium
```
### Install GDAL
```
sudo apt-get install gdal-bin
sudo apt-get install python3-gdal
```
### Install UVC camera controllor
```
sudo apt-get install v4l-utils
```

## Run
```
python3 realtime_waggledecoding_jetson.py 15 --loop 4 # Execute 15 min decoding and repeat 4 times (total 1 h)
```
Use `-h` to see options for setting hive location, threshold value, camera resolution and resize ratio, etc.
