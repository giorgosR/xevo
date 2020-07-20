#include "gtest/gtest.h"

#include "xevo/ga.hpp"
#include "xevo/analytical_functions.hpp"

#include "xtensor/xio.hpp"


TEST(ga, instatiate)
{
  std::array<std::size_t, 2> shape = {20, 2};
  xt::xarray<double> X = xt::zeros<double>(shape);
  
  xevo::ga genetic_algorithm;
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

TEST(ga, void_evolve)
{
  std::array<std::size_t, 2> shape = {40, 2};
  xt::xarray<double> X = xt::zeros<double>(shape);
  
  xevo::Rosenbrock objective_f;

  xevo::ga genetic_algorithm;
  genetic_algorithm.initialise(X);

  std::size_t num_generations = 300;
  for (auto i{0}; i<num_generations; ++i)
  {
    genetic_algorithm.evolve(X, objective_f, std::make_tuple(0.05),
    std::make_tuple(),
    std::make_tuple(0.8), std::make_tuple(0.5, 60.0));
  }
    
  double best_x1 = 0.666;
  double best_x2 = 0.666;

  EXPECT_NEAR(best_x1, X(0,0), 1e-003);
  EXPECT_NEAR(best_x2, X(0,1), 1e-003);

}

TEST(ga, auto_evolve)
{
  std::array<std::size_t, 2> shape = {40, 2};
  xt::xarray<double> X = xt::zeros<double>(shape);
  
  xevo::Rosenbrock objective_f;

  xevo::ga genetic_algorithm;
  genetic_algorithm.initialise(X);

  std::size_t gen_i{0};
  bool run{true};
  while(run != false)
  {
    run = genetic_algorithm.evolve(X, objective_f,
     std::make_tuple(0.05), std::make_tuple(), std::make_tuple(0.8),
      std::make_tuple(0.5, 60.0), std::make_tuple(300, gen_i));
    gen_i += 1;
  }  
  
  double best_x1 = 0.666;
  double best_x2 = 0.666;

  EXPECT_NEAR(best_x1, X(0,0), 1e-003);
  EXPECT_NEAR(best_x2, X(0,1), 1e-003);

}

TEST(ga, auto_evolve_tol)
{
  using xtensor_x_type = xt::xarray<double>;
  using objective_type = xevo::Rosenbrock;
  using elitism_type = xevo::Elitism;
  using selection_type = xevo::Roulette_selection;
  using crossover_type = xevo::Crossover;
  using mutation_type = xevo::Mutation_polynomial;
  using termination_type = xevo::Terminate_tol;

  std::array<std::size_t, 2> shape = {40, 2};
  xtensor_x_type X = xt::zeros<double>(shape);
  
  xevo::Rosenbrock objective_f;

  xevo::ga genetic_algorithm;
  genetic_algorithm.initialise(X);

  auto y = objective_f(X);
  double y_best = xt::flip(xt::sort(y), 0)(0);

  std::size_t gen_i{0};
  std::size_t max_generations{300};
  std::size_t stall{10};
  double tol = std::numeric_limits<double>::max();
  std::size_t stall_i{0};
  while((stall_i < stall) && (gen_i < max_generations))
  {
    auto y_best_n = genetic_algorithm.evolve<xtensor_x_type, objective_type,
     elitism_type, selection_type, crossover_type, mutation_type, termination_type>(X, objective_f,
     std::make_tuple(0.05), std::make_tuple(), std::make_tuple(0.8),
      std::make_tuple(0.5, 60.0), std::make_tuple());
    tol = fabs(y_best_n - y_best);
    y_best = y_best_n;

    if (tol <= 1e-006)
    {
      stall_i += 1;
    }
    else
    {
      stall_i = 0;
    }
    
    gen_i += 1;
  }

  double best_x1 = 0.666;
  double best_x2 = 0.666;

  EXPECT_NEAR(best_x1, X(0,0), 1e-003);
  EXPECT_NEAR(best_x2, X(0,1), 1e-003);

  EXPECT_TRUE((stall_i == stall) && (gen_i < max_generations));
}
