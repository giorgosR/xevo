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

TEST(ga, void_evolve)
{
  std::array<std::size_t, 2> shape = {40, 2};
  xt::xarray<double> X = xt::zeros<double>(shape);
  
  xnio::Rosenbrock objective_f;

  xnio::ga genetic_algorithm;
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
  
  xnio::Rosenbrock objective_f;

  xnio::ga genetic_algorithm;
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
  using objective_type = xnio::Rosenbrock;
  using elitism_type = xnio::Elitism;
  using selection_type = xnio::Roulette_selection;
  using crossover_type = xnio::Crossover;
  using mutation_type = xnio::Mutation_polynomial;
  using termination_type = xnio::Terminate_tol;

  std::array<std::size_t, 2> shape = {40, 2};
  xtensor_x_type X = xt::zeros<double>(shape);
  
  xnio::Rosenbrock objective_f;

  xnio::ga genetic_algorithm;
  genetic_algorithm.initialise(X);

  auto y = objective_f(X);
  double y_best = xt::flip(xt::sort(y), 0)(0);

  std::size_t gen_i{0};
  std::size_t max_generations{300};
  double tol = std::numeric_limits<double>::max();
  while((tol > 1e-009) && (gen_i < max_generations))
  {
    auto y_best_n = genetic_algorithm.evolve<xtensor_x_type, objective_type,
     elitism_type, selection_type, crossover_type, mutation_type, termination_type>(X, objective_f,
     std::make_tuple(0.05), std::make_tuple(), std::make_tuple(0.8),
      std::make_tuple(0.5, 60.0), std::make_tuple());
    tol = fabs(y_best_n - y_best);
    y_best = y_best_n;
    gen_i += 1;
  }
  
  double best_x1 = 0.666;
  double best_x2 = 0.666;

  EXPECT_NEAR(best_x1, X(0,0), 1e-003);
  EXPECT_NEAR(best_x2, X(0,1), 1e-003);

  EXPECT_TRUE((tol<=1e-009) && (gen_i < max_generations));
}