#include "gtest/gtest.h"

#include "xevo/pso_ga.hpp"
#include "xevo/analytical_functions.hpp"

#include "xtensor/xio.hpp"

TEST(pso_ga, void_evolve_sphere)
{
  std::array<std::size_t, 2> shape = { 30, 2 };
  std::array<std::size_t, 1> shape_y = { 30 };

  xt::xarray<double> pop = xt::zeros<double>(shape);

  xevo::Sphere objective_f;

  xevo::pso_ga pso_ga_algorithm;
  pso_ga_algorithm.initialise(pop);
  xt::xarray<double> A(pop);

  xt::xarray<double> Xm1(pop);
  xt::xarray<double> YB = objective_f(A);

  std::size_t num_generations = 200;
  for (auto i{ 0 }; i < num_generations; ++i)
  {
    pso_ga_algorithm.evolve(pop, Xm1, YB, A, objective_f,
      std::make_tuple(0.5, 2.1, 2.1, 20, true), std::make_tuple(true), std::make_tuple(0.00, 50.0));
  }

  auto y_args_sort = xt::argsort(YB);
  auto x_best = xt::view(pop, y_args_sort(0), xt::all());

  double best_x1 = 0.5;
  double best_x2 = 0.5;

  EXPECT_NEAR(best_x1, x_best(0), 1e-006);
  EXPECT_NEAR(best_x2, x_best(1), 1e-006);

}