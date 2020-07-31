[![Build Status](https://dev.azure.com/giorgosragos/giorgosragos/_apis/build/status/giorgosR.xevo?branchName=master)](https://dev.azure.com/giorgosragos/giorgosragos/_build/latest?definitionId=2&branchName=master)

![xevo](doc/images/xevo_logo.png)

# Evolutionary and Swarm Intelligence algorithms

This is the `git` repository for the c++ package xevo.

## Dependencies

`xevo` depends on [`xtensor`](https://github.com/xtensor-stack/xtensor) developed by [`QuantStack`](https://quantstack.net/).

## Clone

To clone the repository type:

```shell
git clone https://github.com/giorgosR/xevo.git
```

## Install

You can install `xevo` from `conda` or build the source with `cmake`.

* Anaconda

```shell
conda install xevo -c giorgosR
```

* CMAKE

Just go to the git repository and type the following:

```shell
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=<install_dir> -DCMAKE_INSTALL_LIBDIR=<lib_dir> ../
cmake --build ./ INSTALL
```

## Docker

To build local docker images with C++ jupyter notebooks, install `docker` from a terminal type:

```bash
docker build -t <image_name> -f .\junbs\Dockerfile .
```

Once built, run the image as

```bash
docker run -p 8080:8888 <image_name>:tag
```

This will start a jupyter kernel inside the docker image. To connect to the kernel, open your browser and paste the following address:

```url
http://localhost:8080
```

Then paste the token which can be found on the docker running image.

## Documentation

To build the documentation, the following packages are needed:

* doxygen
* sphinx
* sphinx_rtd_theme
* breathe

Create a conda environment from [conda yml file](conda/docs.yml) as:

```bash
conda env create -n docs -f docs.yml
conda activate docs
```

Once the environment is created, inside build folder reconfigure as:

```bash
mkdir build && cd build

cmake -G Ninja -DBUILD_DOCUMENTATION=ON
 -DDOXYGEN_EXECUTABLE=${CONDA_PREFIX}\Scripts\doxygen.exe"
 -DSPHINX_EXECUTABLE=${CONDA_PREFIX}\Scripts\sphinx-build.exe" ../

ninja doc && ninja Sphinx
```
