/**
 * @file analytical_functions.hpp
 * @author Georgios E. Ragkousis (giorgosragos@gmail.com)
 * @brief templated header file with functors for analytical functions.
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
#ifndef __ANALYTICAL_FUNCTIONS_HPP__
#define __ANALYTICAL_FUNCTIONS_HPP__

#include <iostream>

#include "xtensor/xexpression.hpp"


namespace xnio
{

  /**
   * @brief branin function modified by Forrester et al. 2006
   *
   * \f[
   *   f(x, y) = a*(X2 - b*X1*X1 + c*X1 - d)**2 + e*(1-f)*\cos{x1} + e)+5*_x1
   * \f] 
   * 
   */
  struct Branin
  {

    /**
     * @brief operator to evaluate the objective function.
     * 
     * @tparam E xtensor type
     * @tparam value_type xtensor value type 
     * @param X array to be evaluated
     * @return auto evaluated array
     */
    template <class E, typename T = typename std::decay_t<E>::value_type>
    inline auto operator()(const xt::xexpression<E>& X)
    {
      double beta = 8.0;
      auto y = evaluate(X);
      auto max_index = xt::argmax(y)();
      T y_max = y(max_index);
      T factor = (-1)*(beta / y_max);
      return xt::eval(xt::exp(factor * (y)));
    }

    /**
     * @brief Get the bounder of Branin functions
     * 
     * @return std::pair<std::vector<double>, std::vector<double>> 
     */
    std::pair<std::vector<double>, std::vector<double>> bounder() const
    {
      return { {-5.0, 0.0}, {10.0, 15.0} };
    }

  private:
    
    /**
     * @brief private method to evaluate branin function
     * 
     * @tparam E xtensor type
     * @tparam value_type value type of E 
     * @param X array to be evaluated
     * @return auto array evaluated at X
     */
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

      auto X1 = 15 * _x1 - 5;
      auto X2 = 15 * _x2;
      value_type a = 1;
      value_type b = 5.1 / (4 * xt::numeric_constants<value_type>::PI*
        xt::numeric_constants<value_type>::PI);
      value_type c = 5 / xt::numeric_constants<value_type>::PI;
      value_type d = 6;
      value_type e = 10;
      value_type f = 1 / (8 * xt::numeric_constants<value_type>::PI);

      xt::xtensor<value_type, 1, xt::layout_type::row_major> y =
        xt::eval(a*xt::pow(X2 - b * X1*X1 + c * X1 - d, 2) + e * (1 - f)*xt::cos(X1) + e) + 5 * _x1;

      return y;
    }

  };

  
  /**
   * @brief Rosenbrock's function.
   *
   * \f[
   *   f(x_1, x_2) = 100*(x_1^2 - x_2)^2 + (1 - x_1)^2 \quad with \quad \mathbf{X} \in \[-3, 3 \]
   * \f]
   * 
   */
  struct Rosenbrock
  {

    /**
     * @brief operator to evaluate the objective function.
     * 
     * @tparam E xtensor type
     * @tparam value_type xtensor value type 
     * @param X array to be evaluated
     * @return auto evaluated array
     */
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

    /**
     * @brief get the bounder of Rosenbrock function
     * 
     * @return std::pair<std::vector<double>, std::vector<double>> 
     */
    std::pair<std::vector<double>, std::vector<double>> bounder() const
    {
       return {{-3, -3}, {3, 3}};
    }

      private:

    /**
     * @brief private method to evaluate branin function
     * 
     * @tparam E xtensor type
     * @tparam value_type value type of E 
     * @param X array to be evaluated
     * @return auto array evaluated at X
     */
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

}

#endif