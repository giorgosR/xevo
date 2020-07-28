/**
 * @file functors.hpp
 * @author Georgios E. Ragkousis (giorgosragos@gmail.com)
 * @brief templated header file for defining (default) functors for evolutionary algorithms.
 * @version @PROJECT_NUMBER
 * @date 2020-07
 * 
 * Distributed under the terms of the BSD 3-Clause License.
 * 
 * The full license is in the file LICENSE, distributed with this software.
 *
 * @copyright Copyright (c) 2020, Georgios E. Ragkousis
 * 
 */
#ifndef __FUNCTORS_H__
#define __FUNCTORS_H__

#include<iostream>
#include <random>
#include <iterator>

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xio.hpp"


namespace xevo
{

  /**
   * @brief functor for generating initial population
   * 
   */
  struct Population
  {
    template <class E, typename T = typename std::decay_t<E>::value_type>
    void operator()(xt::xexpression<E>& X)
    {
      E& _X = X.derived_cast();
      std::size_t num_of_indiv = _X.shape()[0];
      for (std::size_t i{0}; i < num_of_indiv; ++i)
      {
        auto xe = xt::view(_X, i, xt::all());
        individual( xe );  
      } 
    }

    private:
    
    /**
     * @brief method to generate an individual
     * 
     * @tparam E xtensor type 
     * @tparam T value type of E 
     * @param X xtensor array
     */
    template <class E, typename T = typename std::decay_t<E>::value_type>
    void individual(xt::xexpression<E>& X)
    {
      E& _X = X.derived_cast();
      T lower_limit = 0;
      T upper_limit = 1;
      std::size_t num_of_genes = _X.shape()[0];
      uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
      std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
      std::mt19937_64 rng; // random generator
      rng.seed(ss);
      std::uniform_real_distribution<T> unif_dist(lower_limit, upper_limit);

      for (auto i = 0; i < num_of_genes; ++i)
      {
        _X(i) = std::roundf(unif_dist(rng) * 100) / 100.0;
      }
    }
  };
  
  /**
   * @brief Functor for initialising velocity vector to 0
   * 
   */
  struct Velocity_zero
  {
    template <class E, typename T = typename std::decay_t<E>::value_type>
    void operator()(xt::xexpression<E>& V)
    {
      E& _V = V.derived_cast();
      auto shape = _V.shape();
      _V = xt::zeros<T>(shape);
    }
  };

  /**
   * @brief Functor to calculate the velocity at the next iteration
   * 
   * /f[
   *    V_{ij}^{t+1} = \omega V_{ij}^t + c_1 r_1^t \left( pbestX_{ij} - X_{ij}^t \right) + 
   *                    c_2 r_2^t \left( gbestx_j - X_{ij}^t \right)
   * /f]
   * 
   */
  struct Velocity
  {
    Velocity(double w, double c1, double c2) : _w{w},
     _c1{c1}, _c2{c2}
    {

    }

    template <class E, class F, typename T = typename std::decay_t<E>::value_type>
    void operator()(xt::xexpression<E>& X, xt::xexpression<E>& XB,
     xt::xexpression<E>& V, xt::xexpression<F>& YB)
    {
      E& _X = X.derived_cast();
      E& _XB = XB.derived_cast();
      F& _YB = YB.derived_cast();
      E& _V = V.derived_cast();
      auto shape = _X.shape();
      std::array<std::size_t, 1> shape_rand = { shape[0] };
      F r1 = xt::random::rand<double>(shape_rand, 0.0, 1.0);
      F r2 = xt::random::rand<double>(shape_rand, 0.0, 1.0);

      auto index_best = xt::argmin(_YB)();
      auto gx_best = xt::view(_XB, index_best, xt::all());

      for (std::size_t i{0}; i < shape[0]; ++i)
      {
        xt::view(_V, i, xt::all()) = _w*xt::view(_V, i, xt::all()) + _c1*r1(i)*
        (xt::view(_XB - _X, i, xt::all())) + _c2*r2(i)*(gx_best - xt::view(_X, i, xt::all()));
      }

    }

    private:
    double _w;
    double _c1;
    double _c2;
  };

  /**
   * @brief Functor to calculate the velocity with ring topology at the next iteration
   *
   *  /f[
   *    V_{ij}^{t+1} = \omega V_{ij}^t + c_1 r_1^t \left( pbestX_{ij} - X_{ij}^t \right) + 
   *                    c_2 r_2^t \left( ringbestX_{ij} - X_{ij}^t \right)
   *  /f]
   * 
   */
  struct Velocity_ring_topology
  {
    Velocity_ring_topology(double w, double c1, double c2) : _w{w},
     _c1{c1}, _c2{c2}
    {

    }

    template <class E, class F, typename T = typename std::decay_t<E>::value_type>
    void operator()(xt::xexpression<E>& X, xt::xexpression<E>& XB,
     xt::xexpression<E>& V, xt::xexpression<F>& YB)
    {
      E& _X = X.derived_cast();
      E& _XB = XB.derived_cast();
      F& _YB = YB.derived_cast();
      E& _V = V.derived_cast();
      auto shape = _X.shape();
      std::array<std::size_t, 1> shape_rand = { shape[0] };
      F r1 = xt::random::rand<double>(shape_rand, 0.0, 1.0);
      F r2 = xt::random::rand<double>(shape_rand, 0.0, 1.0);

      E gX_best(_XB);

      for (std::size_t i{0}; i < shape[0]; ++i)
      {
        T value_min = std::numeric_limits<T>::max();
        for (std::size_t j{0}; j < 2; ++j)
        {
          int index = i - 1 + j;
          if (index == -1)
          {
            index = shape[0] - 1;
          }
          if (index == shape[0])
          {
            index = 0;
          }
          T _value = _YB(index);
          if (_value < value_min)
          {
            value_min = _value;
            xt::view(gX_best, i, xt::all()) = xt::view(_XB, index, xt::all());
          }

        }
        xt::view(_V, i, xt::all()) = _w*xt::view(_V, i, xt::all()) + _c1*r1(i)*
        (xt::view(_XB - _X, i, xt::all())) + _c2*r2(i)*(xt::view(gX_best, i, xt::all()) - xt::view(_X, i, xt::all()));
      }

    }

    private:
    double _w;
    double _c1;
    double _c2;
  };

  /**
   * @brief Functor for calculating position at t + 1.
   * 
   * \f[ X_{i,j}^(t+1) = X_{i,j}^(t) + V_{i,j}^(t+1) \f]
   * 
   */
  struct Position
  {
    template <class E, typename T = typename std::decay_t<E>::value_type>
    void operator()(xt::xexpression<E>& X, xt::xexpression<E>& V)
    {
      E& _X = X.derived_cast();
      E& _V = V.derived_cast();

      _X = _X + _V;
    }
  };

  struct Selection_best_pso
  {
    template <class E, class F, typename T = typename std::decay_t<E>::value_type>
    void operator()(xt::xexpression<E>& X, xt::xexpression<E>& XB, xt::xexpression<F>& Y,
     xt::xexpression<F>& YB)
    {
      E& _X = X.derived_cast();
      F& _Y = Y.derived_cast();
      E& _XB = XB.derived_cast();
      F& _YB = YB.derived_cast();
      
      auto shape = _X.shape();
      std::size_t no_individuals = shape[0];

      for (std::size_t i{0}; i < no_individuals; ++i)
      {
        if (_Y(i) < _YB(i))
        {
          _YB(i) = _Y(i);
          xt::view(_XB, i, xt::all()) = xt::view(_X, i, xt::all());
        }
      }
    }
  };

  /**
   * @brief functor to calculate individual selection with Roulette method.
   * 
   */
  struct Roulette_selection
  {
    template <class F, class E, typename T = typename std::decay_t<F>::value_type>
    auto operator()(const xt::xexpression<F>& X, const xt::xexpression<E>& Y)
    {
      const E& _Y = Y.derived_cast();
      const F& _X = X.derived_cast();

      auto _shape = _Y.shape();
      std::size_t num_of_individuals = _shape[0];
      E y_norm(_Y);
      T _sum = xt::sum(_Y)();
      E y_args_sort = xt::flip(xt::argsort(_Y), 0);
      E y_sorted = xt::view(_Y, xt::keep(y_args_sort));
      F X_sorted = xt::view(_X, xt::keep(y_args_sort), xt::all());

      for (auto i = 0; i < num_of_individuals; ++i)
      {
        y_norm(i) = y_sorted(i) / _sum;
      }

      E y_cum_fitness = xt::cumsum(y_norm);
      
      std::random_device rd;
      std::mt19937 gen(rd());
      T min_value = 0;
      T max_value = 1;
      std::uniform_real_distribution<T> distribution(min_value, max_value);
      
      F X_out(_X);

      for (std::size_t i{0}; i < num_of_individuals; ++i)
      {
        T random_num = distribution(gen);
        auto iter = std::lower_bound(y_cum_fitness.storage_cbegin(),
         y_cum_fitness.storage_cend(),
          random_num);
        std::size_t dis = 0;

        if (iter == y_cum_fitness.storage_cbegin())
        {
          dis = 0;
        }
        else if (iter == y_cum_fitness.storage_cend())
        {
          dis = std::distance(y_cum_fitness.storage_cbegin(),
           y_cum_fitness.storage_cend()) - 1;
        }
        else
        {
          dis = std::distance(y_cum_fitness.storage_cbegin(), iter);
        }

        xt::view(X_out, i, xt::all()) = xt::view(X_sorted, dis, xt::all()); 

      }

      return X_out;
    }
  };

  /**
   * @brief Cross over functor
   *
   */
  struct Crossover
  {
    /**
     * @brief Constructor
     * 
     * @param crossoverrate cross over rate 
     */
    Crossover(double crossoverrate) :
     _crossover_rate(crossoverrate)
     {
       
     }

    template <class E,
      typename T = typename std::decay_t<E>::value_type>
      auto operator()(const xt::xexpression<E>& X)->E
    {
      double alpha = 0.5;
      const E& _X = X.derived_cast();

      auto shape_X = _X.shape();
      std::size_t num_of_indiv = shape_X[0];
      std::size_t num_of_vars = shape_X[1];
      std::size_t num_crossover = static_cast<std::size_t>(
        floorl(_crossover_rate * num_of_indiv));
      std::size_t _cross_run{};
      if ((num_crossover % 2) == 0)
      {
        _cross_run = num_crossover / 2;
      }
      else
      {
        _cross_run = (num_crossover / 2) + 1;
      }

      std::array<std::size_t, 2> shape_out = { 2 * _cross_run, num_of_vars };
      E _X_out = xt::view(_X, xt::range(0, 2 * _cross_run), xt::all());

      std::array<std::size_t, 1> shape_rand_var = { _cross_run };
      auto random_index = xt::random::randint<std::size_t>(shape_rand_var, 0,
        num_of_vars);

      for (std::size_t i{ 0 }; i < _cross_run; ++i)
      {
        std::size_t x_k_index = 2*i;
        std::size_t y_k_index = 2*i+1;
        T x_k = alpha * _X(y_k_index, random_index(0, i)) + (1 - alpha)*_X(x_k_index, random_index(0, i));
        T y_k = alpha * _X(x_k_index, random_index(0, i)) + (1 - alpha)*_X(y_k_index, random_index(0, i));
        _X_out(x_k_index, random_index(0, i)) = x_k;
        _X_out(y_k_index, random_index(0, i)) = y_k;
      }

      return _X_out;
    }
    private:
    double _crossover_rate; ///< cross over rate
  };


  /**
   * @brief Functor for polynomial mutation 
   *
   * for a given parent solution \f$ p \in \left[ a, b \right] \f$,
   * the mutated solution \f$ p^' \f$ for a particular variable is created
   * for a random number \f$ u \in \left[ 0, 1 \right]\f$
   * 
   * \f[
   *    p^' = 
   *    \begin{cases*}
   *      p + \bar{\delta_L}\left( p - x_i^{(L)} \right), for \quad u \se 0.5, \\
   *      p + \bar{\delta_R}\left( x_i^{(U)} - p \right), for \quad u > 0.5,
   *    \end{cases*}
   * \f]
   * 
   * Then \f$\bar{\delta_L}\f$ and \f$\bar{\delta_R}\f$ are calculated as
   * 
   * \f[
   *   \bar{\delta_L} = (2u)^{1/(1+ \eta_m)} - 1, for \quad u \se 0.5, \\
   *   \bar{\delta_R} = 1 - (2(1-u))^{1/(1+\eta_m)}, for \quad u > 0.5
   * \f]
   * 
   */
  struct Mutation_polynomial
  {
    /**
     * @brief Construct a new Mutation_functor_polynomial object
     * 
     * @param p_m: polynomial mutation probability
     * @param eta_m: index parameter
     */
    Mutation_polynomial(double p_m, double eta_m) : 
    _p_m{p_m}, _eta_m{eta_m}
    {

    }

    template <class E, typename T = typename std::decay_t<E>::value_type>
    auto operator()(const xt::xexpression<E>& X)
    {
      const E& _X = X.derived_cast();
      E out(_X);
      auto shape_x = _X.shape();
      std::size_t num_mutations = shape_x[0];
      std::size_t num_variables = shape_x[1];
      std::array<std::size_t, 1> shape_random = { num_mutations };
      auto random_num = xt::random::randint<std::size_t>(shape_random, 0, num_variables);
      
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<T> distribution(0.0, 1.0);
      
      for (std::size_t i{ 0 }; i < num_mutations; ++i)
      {
        std::size_t rn = random_num(i);

        T p = _X(i, rn);
        T p_n{};
        T u = distribution(gen);
        
        if (u <= 0.5)
        {
          T delta = pow(2*u, 1/(1+_eta_m)) - 1.0;
          p_n = p + delta*(p);
        }
        else
        {
          T delta = 1 - pow(2*(1-u), 1/(1+_eta_m));
          p_n = p + delta*(1.0 - p);
        }

        out(i, rn) = p_n;
      }

      return out;
    }
  private:
    double _p_m; ///< polynomial mutation probability (\f$ p_m = 1/n \f$)
    double _eta_m; ///< index parameter (usually \f$ \eta_m \in \left[ 20, 100 \right] \f$)
  };



  /**
   * @brief Functor for elitism
   *
   */
  struct Elitism
  {
    /**
     * @brief Constructor
     * 
     * @param er elit rate
     */
    Elitism(double er) : _elite_rate(er)
    {

    }

    template <class E, class F,
      typename T = typename std::decay_t<E>::value_type>
      auto operator()(const xt::xexpression<F>& X, const xt::xexpression<E>& Y)->F
    {
      const F& _X = X.derived_cast();
      const E& _Y = Y.derived_cast();

      std::vector<std::size_t> indices;
      auto shape = _X.shape();
      std::size_t no_of_indiv = shape[0];
      std::size_t no_of_vars = shape[1];
      std::size_t no_of_elites = static_cast<std::size_t>(ceil(_elite_rate * no_of_indiv));
      std::array<std::size_t, 2> shape_out = { no_of_elites, no_of_vars };
      F _X_out = xt::zeros<T>(shape_out);
      auto args = xt::argsort(_Y);
      auto iter_back = args.crbegin();
      for (std::size_t i{ 0 }; i < no_of_elites; ++i)
      {
        xt::view(_X_out, i, xt::all()) = xt::view(_X, *iter_back, xt::all());
        ++iter_back;
      }

      return _X_out;
    }
  private:
    double _elite_rate; ///< elit rate
  };

  /**
   * @brief Functor for terminating evolutionary algorithm
   *
   */
  struct Terminate_gen_max
  {
    /**
     * @brief Construct a new Terminate_gen_max object
     * 
     * @param generations maximum number of generations
     * @param index current generation number
     */
    Terminate_gen_max(std::size_t generations, std::size_t index) : 
    _generations{generations}, _index{index}
    {

    }

    template <class E, class F,
      typename T = typename std::decay_t<E>::value_type>
      bool operator()(const xt::xexpression<F>& X, const xt::xexpression<E>& Y)
    {
      const F& _X = X.derived_cast();
      const E& _Y = Y.derived_cast();

      return (_index < _generations);
    }
  private:
    std::size_t _generations; ///< number of generations
    std::size_t _index; ///< generation number
  };


  /**
   * @brief Functor for terminating evolutionary algorithm
   *  by setting a tolerance (y_best_gen - y_best_gen_n <= tol)
   *
   */
  struct Terminate_tol
  {
    template <class E, class F,
      typename T = typename std::decay_t<E>::value_type>
      T operator()(const xt::xexpression<F>& X, const xt::xexpression<E>& Y)
    {
      const F& _X = X.derived_cast();
      const E& _Y = Y.derived_cast();
      T best = _Y(0);
      return best;
    }

  };

}

#endif 