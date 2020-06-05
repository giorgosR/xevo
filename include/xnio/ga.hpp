#ifndef __GA_HPP__
#define __GA_HPP__

#include "xtensor/xtensor.hpp"

#include "functors.hpp"

namespace xnio
{

class ga
{
public:

template<class E, class INDI = Individual, class POP = Population,
 typename T = typename std::decay_t<E>::value_type>
 void initialise(xt::xexpression<E>& X)
 {
   E& _X = X.derived_cast();
   INDI f_indi;
   POP f_pop;
   f_pop(_X, f_indi);
 }

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

 template<class E, class OBJ, class ELIT = Elitism, class SEL = Roulette_selection,
  class CROSS = Crossover,
 class MUT = Mutation_polynomial, typename... ElitArgs, typename... SelArgs, typename... CrossArgs,
  typename... MutArgs, std::size_t... EIs, std::size_t... SIs, std::size_t... CXIs, std::size_t... MIs,
   typename T = typename std::decay_t<E>::value_type>
 void evolve(xt::xexpression<E>& X, OBJ objective_f, std::tuple<ElitArgs...>&& elitargs,
  std::tuple<SelArgs...>&& selargs, std::tuple<CrossArgs...>&& crossargs, std::tuple<MutArgs...>&& mutargs,
   std::index_sequence<EIs...>, std::index_sequence<SIs...>, std::index_sequence<CXIs...>, std::index_sequence<MIs...>)
 {
   E& _pop = X.derived_cast();
   
   ELIT elit_f(std::get<EIs>(std::move(elitargs))...);
   SEL selection_f(std::get<SIs>(std::move(selargs))...);
   CROSS cross_f(std::get<CXIs>(std::move(crossargs))...);
   MUT mutation_f(std::get<MIs>(std::move(mutargs))...);

   auto _Y = objective_f(_pop);

   E pop_selection = selection_f(_pop, _Y);
  

 }  

};

}

#endif