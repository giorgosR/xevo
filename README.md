![xnio](doc/images/xnio_logo.png)

# XNIO (Nature-inspyred optimisation algorithms)

This is the `git` repository for nature inspyred optimisation algorithms.

## Dependencies

`xnio` is based on `xtensor` developed by `QuantStack`.

## Clone

To clone the repository type:

```shell
git clone https://github.com/giorgosR/xnio.git
```

## Install

You can install `xnio` from `conda` or build the source with `cmake`.

* Anaconda

```shell
conda install xnio -c conda-forge 
```

* CMAKE

Just go to the git repository and type the following:

```shell
mkdir build && cd build
cmake -DCMAKE_INSTALL=<your directory> ../
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