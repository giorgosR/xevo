#include "gtest/gtest.h"

#include "xevo/pso.hpp"
#include "xevo/analytical_functions.hpp"

#include "xtensor/xio.hpp"


TEST(pso, initialise)
{
  std::array<std::size_t, 2> shape = {20, 2};
  xt::xarray<double> X = xt::zeros<double>(shape);
  xt::xarray<double> V = xt::ones<double>(shape);

  xevo::pso pso_algorithm;
  pso_algorithm.initialise(X, V);

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
  
  EXPECT_TRUE(xt::allclose(V, xt::zeros<double>(shape)));
}

TEST(ga, void_evolve)
{
  std::array<std::size_t, 2> shape = {40, 2};
  xt::xarray<double> X = xt::zeros<double>(shape);
  xt::xarray<double> V = xt::zeros<double>(shape);
  double max_double_value = std::numeric_limits<double>::max();
  
  xevo::Rosenbrock objective_f;

  xevo::pso pso_algorithm;
  pso_algorithm.initialise(X, V);
  
  xt::xarray<double> XB(X);
  xt::xarray<double> VB(V);
  xt::xarray<double> YB = xt::ones<double>({40})*max_double_value;

  std::size_t num_generations = 300;
  for (auto i{0}; i<num_generations; ++i)
  {
    pso_algorithm.evolve(X, XB, YB, V, VB, objective_f, std::make_tuple(), std::make_tuple());
  }
    
  // double best_x1 = 0.666;
  // double best_x2 = 0.666;

  // EXPECT_NEAR(best_x1, X(0,0), 1e-003);
  // EXPECT_NEAR(best_x2, X(0,1), 1e-003);

}