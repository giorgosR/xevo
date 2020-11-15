Usage
=====

Getting started
---------------

This section covers examples on using the library once installed.

Optimising Rastrigin's function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to find the minimum of Rastrigin's function, a function that is often used to test the genetic algorithm. We will use the `xevo::ga` and we will describe the appropriate steps needed to solve this optimisation problem with the current library.

First, we have to define the functor that will serve as the objective function evaluator. The functor for Rastrigin's function is depicted below:

.. code-block:: cpp

  struct Rastriginsfcn_scaled
  {
    template <class E, typename T = typename std::decay_t<E>::value_type>
    auto operator()(const xt::xexpression<E>& X)
    {
      double beta = 8.0;
      auto y = evaluate(X);
      auto max_index = xt::argmax(y)();
      T y_max = y(max_index);
      T factor = (-1) * (beta / y_max);
      return xt::eval(xt::exp(factor * (y)));
    }

    std::pair<std::vector<double>, std::vector<double>> bounder() const
    {
      return { {-5, -5}, {5, 5} };
    }

  private:
    template <class E, typename T = typename std::decay_t<E>::value_type>
    auto evaluate(const xt::xexpression<E>& X)
    {
      const E& _X = X.derived_cast();

      auto shape = _X.shape();
      std::size_t dim = _X.dimension();
      if (dim != 2)
      {
        throw std::runtime_error("The input array should be of dim 2");
      }
      xt::xtensor<T, 1, xt::layout_type::row_major> _x1(xt::view(_X, xt::all(), 0));
      xt::xtensor<T, 1, xt::layout_type::row_major> _x2(xt::view(_X, xt::all(), 1));

      // scale in [-5, 5]
      auto X1 = 10.0 * _x1 - 5;
      auto X2 = 10.0 * _x2 - 5;

      xt::xtensor<T, 1, xt::layout_type::row_major> y =
        xt::eval(20 + X1 * X1 + X2 * X2 - 10 * (xt::cos(2 * xt::numeric_constants<double>::PI * X1) +
          xt::cos(2 * xt::numeric_constants<double>::PI * X2)));

      return y;
    }
  };


As can be observed, an objective function evaluation functor has to be consisted of an `operator()(const xt::xexpression<E>& X)` method and a `bounder()` method. For this example, we are going to use the default functors for the genetic algorithm that maximise a problem. As such, we are using a scaling (c.f. :eq:`f_scaled`) to minimise Rastrigin's function. The scaling is expressed as:

.. math:: f_{scaled}(x_1, x_2) = e^{-\frac{\beta}{max(f(x_1, x_2))} f(x_1, x_2)}
   :label: f_scaled

The next step is to initialise the population as

.. code-block:: cpp

  std::array<std::size_t, 2> shape = { 40, 2 }; // define the shape of the initial population
  xt::xarray<double> pop = xt::zeros<double>(shape); // declare the population vector

  xevo::ga genetic_algorithm;
  genetic_algorithm.initialise(pop); // initialise the population with genes generated ramdomly in [0, 1].


Lastly, evolve the population in `n` generations:

.. code-block:: cpp

  xevo::Rastriginsfcn_scaled objective_f; // objective function

  std::size_t num_generations = 300; // number of generations
  for (auto i{ 0 }; i < num_generations; ++i)
  {
    genetic_algorithm.evolve(pop, objective_f,
      std::make_tuple(0.05), // arguments for elitism
      std::make_tuple(), // arguments for individual selection
      std::make_tuple(0.8), // arguments for crossover
      std::make_tuple(0.1, 60.0) // arguments for mutation
      );
  }
