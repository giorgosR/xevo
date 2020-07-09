/**
 * @file ga.hpp
 * @author Georgios E. Ragkousis (giorgosragos@gmail.com)
 * @brief Templated header file for genetic algorithm.
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
#ifndef __GA_HPP__
#define __GA_HPP__

#include "xtensor/xtensor.hpp"

#include "functors.hpp"

namespace xnio
{

/**
 * @brief class for genetic algorithm
 * 
 */
class ga
{
public:

/**
 * @brief method to initialise population for ga
 * 
 * @tparam E xtensor type for generating the initial population
 * @tparam POP functor for generating the population
 * @tparam PopArgs optional type of arguments for population functor
 * @tparam T value type of xtensor 
 * @param X array of population
 * @param popargs optional arguments for population functor
 */
template<class E, class POP = Population, typename... PopArgs, 
 typename T = typename std::decay_t<E>::value_type>
 void initialise(xt::xexpression<E>& X, std::tuple<PopArgs...> popargs = std::make_tuple())
 {
   initialise(X, std::move(popargs), std::index_sequence_for<PopArgs...>{});
 }


 /**
  * @brief method to evolve the population
  * 
  * @tparam E xtensor type for population at the current generation.
  * @tparam OBJ functor for objective function
  * @tparam ELIT functor for elitism
  * @tparam SEL functor for selection
  * @tparam CROSS functor for crossover 
  * @tparam MUT functor for mutation
  * @tparam ElitArgs argument types for Elit functor
  * @tparam SelArgs types of arguments for Selection functor
  * @tparam CrossArgs types of arguments for cross over functor
  * @tparam MutArgs types of arguments for mutation functor
  * @tparam T value type of xtensor
  * @param X array with population at current evolution
  * @param objective_f objective function
  * @param elitargs function for elitism
  * @param selargs function for selection
  * @param crossargs function for crossover
  * @param mutargs function for mutation
  */
 template<class E, class OBJ, class ELIT = Elitism, class SEL = Roulette_selection, class CROSS = Crossover,
  class MUT = Mutation_polynomial, typename... ElitArgs, typename... SelArgs, typename... CrossArgs,
  typename... MutArgs, typename T = typename std::decay_t<E>::value_type>
 void evolve(xt::xexpression<E>& X, OBJ objective_f, std::tuple<ElitArgs...> elitargs = std::make_tuple(),
  std::tuple<SelArgs...> selargs = std::make_tuple(), std::tuple<CrossArgs...> crossargs = std::make_tuple(),
   std::tuple<MutArgs...> mutargs = std::make_tuple())
 {
   evolve(X, objective_f, std::move(elitargs), std::move(selargs), std::move(crossargs), std::move(mutargs), 
   std::index_sequence_for<ElitArgs...>{}, std::index_sequence_for<SelArgs...>{}, std::index_sequence_for<CrossArgs...>{},
    std::index_sequence_for<MutArgs...>{});
 }

 private:

 /**
  * @brief private method to initialise population for ga
  * 
  * @tparam E xtensor type for initial population
  * @tparam POP functor for generating the initial population
  * @tparam PopArgs optional type of arguments for population functor
  * @tparam PIs size of population arguments
  * @tparam T value type of xtensor
  * @param X xtensor array for initial population
  * @param popargs 
  */
 template<class E, class POP = Population, typename... PopArgs,  std::size_t... PIs,  
 typename T = typename std::decay_t<E>::value_type>
 void initialise(xt::xexpression<E>& X, std::tuple<PopArgs...>&& popargs, std::index_sequence<PIs...>)
 {
   E& _X = X.derived_cast();
   POP f_pop(std::get<PIs>(std::move(popargs))...);
   f_pop(_X);
 }

 /**
  * @brief method to evolve the population
  * 
  * @tparam E xtensor type for population at the current generation.
  * @tparam OBJ functor for objective function
  * @tparam ELIT functor for elitism
  * @tparam SEL functor for selection
  * @tparam CROSS functor for crossover 
  * @tparam MUT functor for mutation
  * @tparam ElitArgs argument types for Elit functor
  * @tparam SelArgs types of arguments for Selection functor
  * @tparam CrossArgs types of arguments for cross over functor
  * @tparam MutArgs types of arguments for mutation functor
  * @tparam EIs size of arguments for elit functor
  * @tparam SIs size of arguments for selection functor
  * @tparam CXIs size of arguments for crossover functor
  * @tparam MIs size of arguments for mutation functor
  * @tparam T value type of xtensor
  * @param X array with population at current evolution
  * @param objective_f objective function
  * @param elitargs function for elitism
  * @param selargs function for selection
  * @param crossargs function for crossover
  * @param mutargs function for mutation
  */
 template<class E, class OBJ, class ELIT = Elitism, class SEL = Roulette_selection,
  class CROSS = Crossover,
 class MUT = Mutation_polynomial, typename... ElitArgs, typename... SelArgs, typename... CrossArgs,
  typename... MutArgs, std::size_t... EIs, std::size_t... SIs, std::size_t... CXIs, std::size_t... MIs,
   typename T = typename std::decay_t<E>::value_type>
 void evolve(xt::xexpression<E>& X, OBJ objective_f, std::tuple<ElitArgs...>&& elitargs,
  std::tuple<SelArgs...>&& selargs, std::tuple<CrossArgs...>&& crossargs, std::tuple<MutArgs...>&& mutargs,
   std::index_sequence<EIs...>, std::index_sequence<SIs...>, std::index_sequence<CXIs...>, std::index_sequence<MIs...>)
 {

   ELIT elite_f(std::get<EIs>(std::move(elitargs))...);
   SEL selection_f(std::get<SIs>(std::move(selargs))...);
   CROSS cross_f(std::get<CXIs>(std::move(crossargs))...);
   MUT mutation_f(std::get<MIs>(std::move(mutargs))...);

   E& population = X.derived_cast();
   auto y = objective_f(population);

   auto shape_of_population = population.shape();
   std::size_t individual_size = shape_of_population[0];
   std::size_t variable_size = shape_of_population[1];

   //selection
   E population_selection = selection_f(population, y);

   // apply elitism
   E elite_population = elite_f(population, y);
   auto shape_of_elitism = elite_population.shape();
   std::size_t elite_size = shape_of_elitism[0];

   E mating_population = xt::view(population_selection,
     xt::range(elite_size, individual_size));

   // apply crossover
   E population_cross = cross_f(mating_population);
   auto shape_of_cross = population_cross.shape();
   std::size_t cross_size = shape_of_cross[0];

   E mutation_population = xt::view(population_selection,
     xt::range(elite_size + cross_size,
       individual_size), xt::all());

   // apply mutation
   E population_mutated = mutation_f(mutation_population);
   auto shape_of_mutation = population_mutated.shape();
   std::size_t mutation_size = shape_of_mutation[0];

   assert(individual_size == (elite_size + cross_size + mutation_size));

   population = xt::concatenate(xt::xtuple(elite_population,
     population_cross, population_mutated), 0);
 }  

};

}

#endif