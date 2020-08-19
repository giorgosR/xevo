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

TEST(pso, void_evolve_sphere)
{
  std::array<std::size_t, 2> shape = {30, 2};
  std::array<std::size_t, 1> shape_y = {30};

  xt::xarray<double> X = xt::zeros<double>(shape);
  xt::xarray<double> V = xt::zeros<double>(shape);
  double max_double_value = std::numeric_limits<double>::max();
  
  xevo::Sphere objective_f;

  xevo::pso pso_algorithm;
  pso_algorithm.initialise(X, V);
  
  xt::xarray<double> XB(X);
  xt::xarray<double> YB = xt::ones<double>(shape_y)*max_double_value;

  std::size_t num_generations = 100;
  for (auto i{0}; i<num_generations; ++i)
  {
    pso_algorithm.evolve(X, XB, YB, V, objective_f, std::make_tuple(),
     std::make_tuple(0.5, 0.8, 0.9), std::make_tuple());
  }

  auto y_args_sort = xt::argsort(YB); 
  auto x_best = xt::view(XB, y_args_sort(0), xt::all());

  double best_x1 = 0.5;
  double best_x2 = 0.5;

  EXPECT_NEAR(best_x1, x_best(0), 1e-006);
  EXPECT_NEAR(best_x2, x_best(1), 1e-006);

}

TEST(pso, void_evolve_rosenbork)
{
  using xtensor_x_type = xt::xarray<double>;
  using objective_type = xevo::Rosenbrock;
  using population_type = xevo::Population; 

  std::array<std::size_t, 2> shape = {40, 2};
  std::array<std::size_t, 1> shape_y = {40};

  xt::xarray<double> X = xt::zeros<double>(shape);
  xt::xarray<double> V = xt::zeros<double>(shape);
  double max_double_value = std::numeric_limits<double>::max();
  
  xevo::Rosenbrock objective_f;

  xevo::pso pso_algorithm;
  pso_algorithm.initialise<xtensor_x_type,
   population_type, population_type>(X, V);
  
  xt::xarray<double> XB(X);
  xt::xarray<double> YB = xt::ones<double>(shape_y)*max_double_value;

  std::size_t num_generations = 300;
  for (auto i{0}; i<num_generations; ++i)
  {
    pso_algorithm.evolve(X, XB, YB, V, objective_f, std::make_tuple(),
     std::make_tuple(0.5, 0.8, 0.9), std::make_tuple());
  }

  auto y_args_sort = xt::argsort(YB); 
  auto x_best = xt::view(XB, y_args_sort(0), xt::all());

  double best_x1 = 0.666;
  double best_x2 = 0.666;

  EXPECT_NEAR(best_x1, x_best(0), 1e-003);
  EXPECT_NEAR(best_x2, x_best(1), 1e-003);

}


TEST(pso, void_evolve_rosenbork_ring_topology)
{
  using xtensor_x_type = xt::xarray<double>;
  using objective_type = xevo::Rosenbrock;
  using population_type = xevo::Population;
  using selection_type = xevo::Selection_best_pso;
  using velocity_type = xevo::Velocity_ring_topology;
  using position_type = xevo::Position;

  std::array<std::size_t, 2> shape = {50, 2};
  std::array<std::size_t, 1> shape_y = {50};

  xt::xarray<double> X = xt::zeros<double>(shape);
  xt::xarray<double> V = xt::zeros<double>(shape);
  double max_double_value = std::numeric_limits<double>::max();
  
  xevo::Rosenbrock objective_f;

  xevo::pso pso_algorithm;
  pso_algorithm.initialise<xtensor_x_type,
   population_type, population_type>(X, V);
  
  xt::xarray<double> XB(X);
  xt::xarray<double> YB = xt::ones<double>(shape_y)*max_double_value;

  std::size_t num_generations = 300;
  for (auto i{0}; i<num_generations; ++i)
  {
    pso_algorithm.evolve<xtensor_x_type, xtensor_x_type, objective_type,
     position_type, velocity_type, selection_type>(X, XB, YB, V, objective_f, std::make_tuple(),
     std::make_tuple(0.5, 0.8, 0.9), std::make_tuple());
  }

  auto y_args_sort = xt::argsort(YB); 
  auto x_best = xt::view(XB, y_args_sort(0), xt::all());

  double best_x1 = 0.666;
  double best_x2 = 0.666;

  EXPECT_NEAR(best_x1, x_best(0), 1e-003);
  EXPECT_NEAR(best_x2, x_best(1), 1e-003);

}