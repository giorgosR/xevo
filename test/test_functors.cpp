#include "gtest/gtest.h"

#include "xtensor/xio.hpp"

#include "xnio/functors.hpp"

TEST(functors, invividual)
{
  std::array<std::size_t, 1> shape = {2};
  xt::xarray<double> X = xt::zeros<double>(shape);
    
  xnio::Individual indi_f;
  indi_f(X);

  double magn = xt::pow(xt::sum(X*X), 0.5)();

  bool is_greater_zero = magn >= 1e-015;  

  EXPECT_TRUE(is_greater_zero); 
}

TEST(functors, population)
{
  std::array<std::size_t, 2> shape = {20, 2};
  xt::xarray<double> X = xt::zeros<double>(shape);
    
  xnio::Individual indi_f;
  xnio::Population pop_f;

  pop_f(X, indi_f);

  bool is_between_limits = true;  
  
  for (std::size_t i{0}; i < 20; ++i)
  {
    double magn = xt::pow(xt::sum(xt::view(X, i, xt::all())*xt::view(X, i, xt::all())), 0.5)();
    is_between_limits = magn >= 1e-015;
    if (!is_between_limits)
    {
      is_between_limits = false;
      break;
    }
  }

  EXPECT_TRUE(is_between_limits); 
}