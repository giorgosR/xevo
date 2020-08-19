/**
 * @file pso.hpp
 * @author Georgios E. Ragkousis (giorgosragos@gmail.com)
 * @brief Templated header file for particle swarm optimisation algorithm.
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
#ifndef __PSO_H__
#define __PSO_H__

#include "xtensor/xtensor.hpp"

#include "functors.hpp"


namespace xevo
{

class pso
{
public:

/**
 * @brief method to initialise position and velocity of swarm for pso
 * 
 * @tparam E xtensor type for position and velocity initial vectors 
 * @tparam POS functor type for generating the initial positions
 * @tparam VEL functor type for generating the initial velocity
 * @tparam PosArgs type of arguments for initialising position functor
 * @tparam VelArgs type of arguments for initialising velocity functor
 * @tparam T value type of xtensor 
 * @param X array of initial bird positions
 * @param V array of initial bird velocities
 * @param posargs (optional) arguments for position functor
 * @param velargs (optional) arguments for velocity functor
 */
template<class E, class POS = Population, class VEL = Velocity_zero, typename... PosArgs, 
 typename... VelArgs, typename T = typename std::decay_t<E>::value_type>
 void initialise(xt::xexpression<E>& X, xt::xexpression<E>& V, std::tuple<PosArgs...> posargs = std::make_tuple(),
  std::tuple<VelArgs...> velargs = std::make_tuple())
 {
   initialise<E, POS, VEL>(X, V, std::move(posargs), std::move(velargs), std::index_sequence_for<PosArgs...>{},
    std::index_sequence_for<VelArgs...>{});
 }

 /**
  * @brief  method to evolve bird positions of the swarm
  * 
  * @tparam E xtensor type for input and output vectors
  * @tparam F xtensor type for input and output vectors for evaluation best
  * @tparam OBJ Functor type for objective function evaluation
  * @tparam POS Functor type for position evaluation
  * @tparam VEL Functor type for velocity evaluation
  * @tparam PosArgs type of arguments for position evaluation functor
  * @tparam VelArgs type of arguments for velocity evaluation functor
  * @tparam T value type of xtensor
  * @param X vector with initial positions of the swarm
  * @param XB vector with best positions of the individuals comprising the swarm
  * @param YB vector with best evaluations of the individuals comprising the swarm
  * @param V vector with initial velocities of the swarm individuals
  * @param objective_f functor for objective function evaluation
  * @param posargs tuple with arguments for position functor
  * @param velargs tuple with arguments for velocity functor
  */
 template<class E, class F, class OBJ, class POS = Position, class VEL=Velocity, class SEL=Selection_best_pso,
  typename... PosArgs, typename... VelArgs, typename... SelArgs, typename T = typename std::decay_t<E>::value_type>
 void evolve(xt::xexpression<E>& X, xt::xexpression<E>& XB, xt::xexpression<F>& YB,
  xt::xexpression<E>& V,
  OBJ objective_f, std::tuple<PosArgs...> posargs, std::tuple<VelArgs...> velargs,
   std::tuple<SelArgs...> selargs)
 {
   evolve<E, F, OBJ, POS, VEL, SEL>(X, XB, YB, V, objective_f, std::move(posargs), std::move(velargs), 
   std::move(selargs), std::index_sequence_for<PosArgs...>{}, std::index_sequence_for<VelArgs...>{},
    std::index_sequence_for<SelArgs...>{});
 }


 /**
  * @brief  method to evolve bird positions of the swarm
  * 
  * @tparam E xtensor type for input and output vectors
  * @tparam F xtensor type for input and output vectors for evaluation best
  * @tparam OBJ Functor type for objective function evaluation
  * @tparam POS Functor type for position evaluation
  * @tparam VEL Functor type for velocity evaluation
  * @tparam SEL Functor type for selection best evaluation
  * @tparam TERM Functor type for termination of pso
  * @tparam PosArgs type of arguments for position evaluation functor
  * @tparam VelArgs type of arguments for velocity evaluation functor
  * @tparam SelArgs type of arguments for best selection evaluation functor
  * @tparam T value type of xtensor
  * @param X vector with initial positions of the swarm
  * @param XB vector with best positions of the individuals comprising the swarm
  * @param YB vector with best evaluations of the individuals comprising the swarm
  * @param V vector with initial velocities of the swarm individuals
  * @param objective_f functor for objective function evaluation
  * @param posargs tuple with arguments for position functor
  * @param velargs tuple with arguments for velocity functor
  * @param selargs tuple with arguments for selection functor
  * @param termargs tuple with arguments for termination functor
  * 
  */
 template<class E, class F, class OBJ, class POS = Position, class VEL=Velocity, class SEL=Selection_best_pso,
  class TERM=Terminate_gen_max,
  typename... PosArgs, typename... VelArgs, typename... SelArgs, typename... TermArgs,
   typename T = typename std::decay_t<E>::value_type>
 auto evolve(xt::xexpression<E>& X, xt::xexpression<E>& XB, xt::xexpression<F>& YB,
  xt::xexpression<E>& V,
  OBJ objective_f, std::tuple<PosArgs...> posargs, std::tuple<VelArgs...> velargs,
   std::tuple<SelArgs...> selargs, std::tuple<TermArgs...> termargs)
 {
   return evolve<E, F, OBJ, POS, VEL, SEL, TERM>(X, XB, YB, V, objective_f, std::move(posargs), std::move(velargs), 
   std::move(selargs), std::move(termargs), std::index_sequence_for<PosArgs...>{}, std::index_sequence_for<VelArgs...>{},
    std::index_sequence_for<SelArgs...>{}, std::index_sequence_for<TermArgs...>{});
 }

 private:

/**
 * @brief method to initialise position and velocity of swarm for pso
 * 
 * @tparam E xtensor type for position and velocity initial vectors 
 * @tparam POS functor type for generating the initial positions
 * @tparam VEL functor type for generating the initial velocity
 * @tparam PopArgs type of arguments for initialising position functor
 * @tparam VelArgs type of arguments for initialising velocity functor
 * @tparam PIs number of arguments for initial position functor
 * @tparam VIs number of arguments for initial velocity functor
 * @tparam T value type of xtensor 
 * @param X array of initial bird positions
 * @param V array of initial bird velocities
 * @param popargs (optional) arguments for position functor
 * @param popargs (optional) arguments for velocity functor
 */
 template<class E, class POS = Population, class VEL = Velocity_zero, typename... PosArgs,
  typename... VelArgs,  std::size_t... PIs, std::size_t... VIs,  
 typename T = typename std::decay_t<E>::value_type>
 void initialise(xt::xexpression<E>& X, xt::xexpression<E>& V, std::tuple<PosArgs...>&& posargs,
 std::tuple<VelArgs...>&& velargs, std::index_sequence<PIs...>, std::index_sequence<VIs...>)
 {
   E& _X = X.derived_cast();
   E& _V = V.derived_cast();
   
   POS f_pos(std::get<PIs>(std::move(posargs))...);
   f_pos(_X);
 
   VEL f_vel(std::get<VIs>(std::move(velargs))...);
   f_vel(_V);
 }

 /**
  * @brief  method to evolve bird positions of the swarm
  * 
  * @tparam E xtensor type for input and output vectors
  * @tparam F xtensor type for input and output of evaluation best
  * @tparam OBJ Functor type for objective function evaluation
  * @tparam POS Functor type for position evaluation
  * @tparam VEL Functor type for velocity evaluation
  * @tparam SEL Functor type for selection evaluation (Y best and X best)
  * @tparam PosArgs type of arguments for position evaluation functor
  * @tparam VelArgs type of arguments for velocity evaluation functor
  * @tparam SelArgs type of arguments for selection evaluation functor
  * @tparam PIs number of arguments for position evaluation functor
  * @tparam VIs number of arguments for velocity evaluation functor
  * @tparam SIs number of arguments for selection evaluation functor
  * @tparam T value type of xtensor
  * @param X vector with initial positions of the swarm
  * @param XB vector with best positions of the individuals comprising the swarm
  * @param YB vector with best evaluations of the individuals comprising the swarm
  * @param V vector with initial velocities of the swarm individuals
  * @param objective_f functor for objective function evaluation
  * @param posargs tuple with arguments for position functor
  * @param velargs tuple with arguments for velocity functor
  * @param selargs tuple with arguments for selection functor
  */
 template<class E, class F, class OBJ, class POS = Position, class VEL = Velocity, class SEL = Selection_best_pso,
  typename... PosArgs, typename... VelArgs, typename... SelArgs, std::size_t... PIs, std::size_t... VIs, std::size_t... SIs,
   typename T = typename std::decay_t<E>::value_type>
 void evolve(xt::xexpression<E>& X, xt::xexpression<E>& XB, xt::xexpression<F>& YB,
  xt::xexpression<E>& V, OBJ objective_f, std::tuple<PosArgs...>&& posargs,
  std::tuple<VelArgs...>&& velargs, std::tuple<SelArgs...>&& selargs, std::index_sequence<PIs...>,
   std::index_sequence<VIs...>, std::index_sequence<SIs...>)
 {

   POS pos_f(std::get<PIs>(std::move(posargs))...);
   VEL vel_f(std::get<VIs>(std::move(velargs))...);
   SEL sel_f(std::get<SIs>(std::move(selargs))...);

   E& position = X.derived_cast();
   E& position_best = XB.derived_cast();
   F& y_best = YB.derived_cast();
   E& velocity = V.derived_cast();

   auto shape_of_population = position.shape();
   std::size_t individual_size = shape_of_population[0];
   std::size_t variable_size = shape_of_population[1];

   F y = objective_f(position);

   sel_f(position, position_best, y, y_best);
   
   vel_f(position, position_best, velocity, y_best);

   pos_f(position, velocity);

 }


 /**
  * @brief  method to evolve bird positions of the swarm
  * 
  * @tparam E xtensor type for input and output vectors
  * @tparam F xtensor type for input and output of evaluation best
  * @tparam OBJ Functor type for objective function evaluation
  * @tparam POS Functor type for position evaluation
  * @tparam VEL Functor type for velocity evaluation
  * @tparam SEL Functor type for selection evaluation (Y best and X best)
  * @tparam TERM Functor type for termination
  * @tparam PosArgs type of arguments for position evaluation functor
  * @tparam VelArgs type of arguments for velocity evaluation functor
  * @tparam SelArgs type of arguments for selection evaluation functor
  * @tparam TermArgs type of arguments for termination functor
  * @tparam PIs number of arguments for position evaluation functor
  * @tparam VIs number of arguments for velocity evaluation functor
  * @tparam SIs number of arguments for selection evaluation functor
  * @tparam TIs number of arguments for termination functor
  * @tparam T value type of xtensor
  * @param X vector with initial positions of the swarm
  * @param XB vector with best positions of the individuals comprising the swarm
  * @param YB vector with best evaluations of the individuals comprising the swarm
  * @param V vector with initial velocities of the swarm individuals
  * @param objective_f functor for objective function evaluation
  * @param posargs tuple with arguments for position functor
  * @param velargs tuple with arguments for velocity functor
  * @param selargs tuple with arguments for selection functor
  * @param termargs tuple with arguments for termination functor
  */
 template<class E, class F, class OBJ, class POS = Position, class VEL = Velocity, class SEL = Selection_best_pso,
  class TERM = Terminate_gen_max,
  typename... PosArgs, typename... VelArgs, typename... SelArgs, typename... TermArgs,
   std::size_t... PIs, std::size_t... VIs, std::size_t... SIs, std::size_t... TIs,
   typename T = typename std::decay_t<E>::value_type>
 auto evolve(xt::xexpression<E>& X, xt::xexpression<E>& XB, xt::xexpression<F>& YB,
  xt::xexpression<E>& V, OBJ objective_f, std::tuple<PosArgs...>&& posargs,
  std::tuple<VelArgs...>&& velargs, std::tuple<SelArgs...>&& selargs, std::tuple<TermArgs...>&& termargs,
   std::index_sequence<PIs...>, std::index_sequence<VIs...>, std::index_sequence<SIs...>, std::index_sequence<TIs...>)
 {

   POS pos_f(std::get<PIs>(std::move(posargs))...);
   VEL vel_f(std::get<VIs>(std::move(velargs))...);
   SEL sel_f(std::get<SIs>(std::move(selargs))...);
   TERM terminate_f(std::get<TIs>(std::move(termargs))...);

   E& position = X.derived_cast();
   E& position_best = XB.derived_cast();
   F& y_best = YB.derived_cast();
   E& velocity = V.derived_cast();

   auto shape_of_population = position.shape();
   std::size_t individual_size = shape_of_population[0];
   std::size_t variable_size = shape_of_population[1];

   F y = objective_f(position);

   sel_f(position, position_best, y, y_best);
   
   vel_f(position, position_best, velocity, y_best);

   pos_f(position, velocity);
   
   return terminate_f(position, objective_f(position));
 }

};
}

#endif