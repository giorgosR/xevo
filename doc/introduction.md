Introduction
============

`xevo` is a C++ templated library for creating flexible evolutionary and swarm intelligence algorithms. 
`xevo` currently depends on [`xtensor`](https://github.com/xtensor-stack/xtensor) developed by [`QuantStack`](https://quantstack.net/). `xtensor` is a C++ templated library that offers multi-dimensional arrays for numerical analysis (it has been inspired by [`Numpy`](numpy.org)).

The implemented algorithms of `xevo` are designed in a way so the user can control/define the algorithm's main functional elements (e.g. mutation, crossover, etc.). The classes that express the algorithms are consisted of two variadic templated methods:

1. an initialiser (`Algorithm::initialise(X)`)

   This method is used for initialising population, velocity, position and other vectors that will be evolved in the next generations

2. an evolution (`Algorithm::evolve(X, ...)`)

   This method is used to evolve the population in the next generation

`xevo` requires a modern C++ compiler supporting C++14. The following C++ compilers are supported and tested:

* On Windows platforms, Visual C++ 2017 or more recent.

* On Unix platforms, Clang++6.0 or more recent.

Design concepts
---------------

Each evolutionary/swarm class consists of two variadic templated methods, an initialiser and an evolution method. For instance, the main body of the `xevo::ga` algorithm is attached below

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

Clone
-----

To clone the repository type:

```shell
git clone https://github.com/giorgosR/xevo.git
```

Install
-------

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
