![xevo](doc/images/xevo_logo.png)

[![Build Status](https://dev.azure.com/giorgosragos/giorgosr-xevo/_apis/build/status/giorgosR.xevo?branchName=master)](https://dev.azure.com/giorgosragos/giorgosr-xevo/_build/latest?definitionId=3&branchName=master)
[![Documentation Status](https://readthedocs.org/projects/xevo/badge/?version=latest)](https://xevo.readthedocs.io/en/latest/?badge=latest)

# Evolutionary and Swarm Intelligence algorithms

This is the `git` repository for the c++ package xevo.

## Dependencies

`xevo` depends on [`xtensor`](https://github.com/xtensor-stack/xtensor) developed by [`QuantStack`](https://quantstack.net/).

## Introduction

`xevo` is a C++ templated library for creating flexible evolutionary and swarm intelligence algorithms. The implemented algorithms are designed in a way so the user can control/define the algorithm's main functional elements (e.g. mutation, crossover, etc.).

## Design concepts

The main body of ga algorithm is attached below

```cpp
class ga
{
public:
template<class E, class POP = Population, typename... PopArgs, 
 typename T = typename std::decay_t<E>::value_type>
 void initialise(xt::xexpression<E>& X, std::tuple<PopArgs...> popargs = std::make_tuple());

 template<class E, class OBJ, class ELIT = Elitism, class SEL = Roulette_selection, class CROSS = Crossover,
  class MUT = Mutation_polynomial, typename... ElitArgs, typename... SelArgs, typename... CrossArgs,
  typename... MutArgs, typename T = typename std::decay_t<E>::value_type>
 void evolve(xt::xexpression<E>& X, OBJ objective_f, std::tuple<ElitArgs...> elitargs,
  std::tuple<SelArgs...> selargs, std::tuple<CrossArgs...> crossargs, std::tuple<MutArgs...> mutargs);
};
```

* Initialise

```cpp
<class E, class POP = Population>
xevo::ga::initialise<E, POP>(E& X, std::tuple<PopArgs>(...) popargs)
```

This is a method to initialise the initial population stored in `X` array. The user can provide a Population functor so as to control how the initialisation will be calculated.

* evolve

```cpp
template<class E, class OBJ, class ELIT = Elitism, class SEL = Roulette_selection, class CROSS = Crossover,
  class MUT = Mutation_polynomial>
void evolve(E& X, OBJ objective_f, std::tuple<ElitArgs...> elitargs,
  std::tuple<SelArgs...> selargs, std::tuple<CrossArgs...> crossargs, std::tuple<MutArgs...> mutargs)
```

This method is for evolving the population `X`. The user can provide functors for objective function evaluation, elitism, selection, mutation and crossover calculations.

## Example

Optimise Rosenbrock function with `xevo::ga`.

Rosenbrock function is expressed as:

![rosenbrock](https://render.githubusercontent.com/render/math?math=f(x_1,%20x_2)%20=%20100(x_1^2%20-%20x_2)%20+%20(1%20-%20x_1)^2%20\quad%20with%20\quad%20\bf{X}%20\quad%20\in%20\left[-3,%203\right])

The first step is to provide the objective function as (we use scaling):

![rosenbrock_scaled](https://render.githubusercontent.com/render/math?math=f_{scaled}(x_1,%20x_2)%20=e^{\frac{-1*\beta}{max(f)}})

```cpp
struct Rosenbrock_scaled
{
  template <class E, typename T = typename std::decay_t<E>::value_type>
  auto operator()(const xt::xexpression<E>& X)
  {
    double beta = 8.0;
    auto y = evaluate(X);
    auto max_index = xt::argmax(y)();
    T y_max = y(max_index);
    T factor = (-1)*(beta / y_max);
    return xt::eval(xt::exp(factor * (y)));
  }

private:
  template <class E, typename value_type = typename std::decay_t<E>::value_type>
  inline auto evaluate(const xt::xexpression<E>& X)
  {
    const E& _X = X.derived_cast();

    auto shape = _X.shape();
    std::size_t dim = _X.dimension();
    if (dim != 2)
    {
      throw std::runtime_error("The input array should be of dim 2");
    }
    xt::xtensor<value_type, 1, xt::layout_type::row_major> _x1(xt::view(_X, xt::all(), 0));
    xt::xtensor<value_type, 1, xt::layout_type::row_major> _x2(xt::view(_X, xt::all(), 1));

    // scale in [-3, 3]
    auto X1 = 6.0*_x1 - 3;
    auto X2 = 6.0*_x2 - 3;

    xt::xtensor<value_type, 1, xt::layout_type::row_major> y =
      xt::eval(100.0*xt::pow(xt::pow(X1, 2) - X2, 2) + xt::pow(1 - X1, 2));

    return y;
  }

};
```

Once the objective is defined, we can simply run the ga with the default functors as:

```cpp
std::array<std::size_t, 2> shape = {40, 2};
xt::xarray<double> X = xt::zeros<double>(shape); //define the dimension of X (population size 40 and gene size 2.)

xevo::Rosenbrock_scaled objective_f; // define objective function

xevo::ga genetic_algorithm;
genetic_algorithm.initialise(X); // initialise population with the default population functor (random generation between 0-1)

std::size_t num_generations = 300;

for (auto i{0}; i<num_generations; ++i)
{
    // evolve the population with default functors for elitism, selection, crossover and mutation.
    genetic_algorithm.evolve(X, objective_f, std::make_tuple(0.05),
                          std::make_tuple(), std::make_tuple(0.8),
                          std::make_tuple(0.5, 60.0));
}
```

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
