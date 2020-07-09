#include "gtest/gtest.h"

#include "xtensor/xio.hpp"

#include "xnio/functors.hpp"

TEST(functors, population)
{
  std::array<std::size_t, 2> shape = {20, 2};
  xt::xarray<double> X = xt::zeros<double>(shape);

  xnio::Population pop_f;

  pop_f(X);

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