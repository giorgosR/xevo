#include "gtest/gtest.h"

#include "xnio/ga.hpp"
#include "xnio/analytical_functions.hpp"

#include "xtensor/xio.hpp"


TEST(ga, instatiate)
{
  std::array<std::size_t, 2> shape = {20, 2};
  xt::xarray<double> X = xt::zeros<double>(shape);
  
  xnio::ga genetic_algorithm;
  genetic_algorithm.initialise(X);

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

TEST(ga, evolve)
{
  std::array<std::size_t, 2> shape = {40, 2};
  xt::xarray<double> X = xt::zeros<double>(shape);
  
  xnio::Rosenbrock objective_f;

  xnio::ga genetic_algorithm;
  genetic_algorithm.initialise(X);

  std::cout << "Inital pop: \n" << X << "\n" << std::endl;
  std::cout << "Inital Y: \n" << objective_f(X) << "\n" << std::endl;

  std::size_t num_generations = 300;
  for (auto i{0}; i<num_generations; ++i)
  {
    genetic_algorithm.evolve(X, objective_f, std::make_tuple(0.05),
    std::make_tuple(),
    std::make_tuple(0.8), std::make_tuple(0.5, 60.0));
  }
  
  std::cout << "Last pop: \n" << X << "\n" << std::endl;
  
  double best_x1 = 0.666;
  double best_x2 = 0.666;

  EXPECT_NEAR(best_x1, X(0,0), 1e-003);
  EXPECT_NEAR(best_x2, X(0,1), 1e-003);

}