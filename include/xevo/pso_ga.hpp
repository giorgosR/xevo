/**
 * @file pso_ea.hpp
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
#ifndef __PSO_GA_H__
#define __PSO_GA_H__

#include "xtensor/xtensor.hpp"

#include "functors.hpp"


namespace xevo
{
  /**
   * @brief class for hybrid pso (pso ea) as presented by Deb et al. 2010 (https://dl.acm.org/doi/10.1145/1830483.1830492) 
   * 
   * The position is updated as
   * 
   * \f[ 
   *   \mathbf{x}_i^{t + 1} = \mathbf{x}_i^{t} + w \left( \mathbf{x}_i^{t} - \mathbf{x}_i^{t-1} \right) +
   *                          c_1 r_1 \left( \mathbf{p}_{b, i}^t - \mathbf{x}_i^{t} \right) +
   *                          c_2 r_2 \left( \mathbf{p}_g^t - \mathbf{x}_i^{t} \right)
   * \f]
   * 
   * At the end the user can mutate the population by passing a mutation functor.
   * 
   */
  class pso_ga
  {
  public:

    /**
     * @brief method to initialise position and velocity of swarm for pso
     *
     * @tparam E xtensor type for position and velocity initial vectors
     * @tparam POS functor type for generating the initial positions
     * @tparam PosArgs type of arguments for initialising position functor
     * @tparam T value type of xtensor
     * @param X array of initial bird positions
     * @param posargs (optional) arguments for position functor
     */
    template<class E, class POS = Population, typename... PosArgs,
      typename T = typename std::decay_t<E>::value_type>
      void initialise(xt::xexpression<E>& X, std::tuple<PosArgs...> posargs = std::make_tuple())
    {
      initialise<E, POS>(X, std::move(posargs), std::index_sequence_for<PosArgs...>{});
    }

    /**
     * @brief  method to evolve bird positions of the swarm
     *
     * @tparam E xtensor type for input and output vectors
     * @tparam F xtensor type for input and output vectors for evaluation best
     * @tparam OBJ Functor type for objective function evaluation
     * @tparam POS Functor type for position evaluation
     * @tparam SEL Functor type for selection evaluation
     * @tparam MUT Functor type for mutation evaluation
     * @tparam PosArgs type of arguments for position evaluation functor
     * @tparam SelArgs type of arguments for selection evaluation functor
     * @tparam MutArgs type of arguments for mutation evaluation functor
     * @tparam T value type of xtensor
     * @param X vector with initial positions of the swarm
     * @param Xm1 vector with previous positions (one generation back) of the individuals comprising the swarm
     * @param YB vector with best evaluations of the individuals comprising the swarm
     * @param A vector with archived positions of the swarm individuals
     * @param objective_f functor for objective function evaluation
     * @param posargs tuple with arguments for position functor
     * @param selargs tuple with arguments for velocity functor
     * @param mutargs tuple with arguments for mutation functor
     */
    template<class E, class F, class OBJ, class POS = Position_pso_ga, class SEL = Selection_best_pso_ga,
      class MUT = Mutation_polynomial, typename... PosArgs, typename... SelArgs, typename... MutArgs,
      typename T = typename std::decay_t<E>::value_type>
      void evolve(xt::xexpression<E>& X, xt::xexpression<E>& Xm1, xt::xexpression<F>& YB,
        xt::xexpression<E>& A,
        OBJ objective_f, std::tuple<PosArgs...> posargs,
        std::tuple<SelArgs...> selargs, std::tuple<MutArgs...> mutargs)
    {
      evolve<E, F, OBJ, POS, SEL, MUT>(X, Xm1, YB, A, objective_f, std::move(posargs),
        std::move(selargs), std::move(mutargs), std::index_sequence_for<PosArgs...>{},
        std::index_sequence_for<SelArgs...>{}, std::index_sequence_for<MutArgs...>{});
    }

    /**
     * @brief  method to evolve bird positions of the swarm
     *
     * @tparam E xtensor type for input and output vectors
     * @tparam F xtensor type for input and output vectors for evaluation best
     * @tparam OBJ Functor type for objective function evaluation
     * @tparam POS Functor type for position evaluation
     * @tparam SEL Functor type for selection evaluation
     * @tparam MUT Functor type for mutation evaluation
     * @tparam TERM Functor type for termination of pso
     * @tparam PosArgs type of arguments for position evaluation functor
     * @tparam SelArgs type of arguments for best selection evaluation functor
     * @tparam MutArgs type of arguments for velocity evaluation functor
     * @tparam TermArgs type of arguments for termination evaluation functor
     * @tparam T value type of xtensor
     * @param X vector with initial positions of the swarm
     * @param Xm1 vector with previous positions (one generation back) of the individuals comprising the swarm
     * @param YB vector with best evaluations of the individuals comprising the swarm
     * @param A vector with archived individuals of the swarm individuals
     * @param objective_f functor for objective function evaluation
     * @param posargs tuple with arguments for position functor
     * @param selargs tuple with arguments for selection functor
     * @param mutargs tuple with arguments for mutation functor
     * @param termargs tuple with arguments for termination functor
     *
     */
    template<class E, class F, class OBJ, class POS = Position_pso_ga, class SEL = Selection_best_pso_ga,
      class MUT = Mutation_functor_polynomial, class TERM = Terminate_gen_max,
      typename... PosArgs, typename... SelArgs, typename... MutArgs, typename... TermArgs,
      typename T = typename std::decay_t<E>::value_type>
      auto evolve(xt::xexpression<E>& X, xt::xexpression<E>& Xm1, xt::xexpression<F>& YB,
        xt::xexpression<E>& A,
        OBJ objective_f, std::tuple<PosArgs...> posargs,
        std::tuple<SelArgs...> selargs, std::tuple<MutArgs...> mutargs, std::tuple<TermArgs...> termargs)
    {
      return evolve<E, F, OBJ, POS, SEL, MUT, TERM>(X, Xm1, YB, A, objective_f, std::move(posargs),
        std::move(selargs), std::move(mutargs), std::move(termargs), std::index_sequence_for<PosArgs...>{},
        std::index_sequence_for<SelArgs...>{}, std::index_sequence_for<MutArgs...>{},
        std::index_sequence_for<TermArgs...>{});
    }


  private:

    /**
     * @brief method to initialise position and velocity of swarm for pso
     *
     * @tparam E xtensor type for position and velocity initial vectors
     * @tparam POS functor type for generating the initial positions
     * @tparam PopArgs type of arguments for initialising position functor
     * @tparam PIs number of arguments for initial position functor
     * @tparam T value type of xtensor
     * @param X array of initial bird positions
     * @param popargs (optional) arguments for position functor
     */
    template<class E, class POS = Population, typename... PosArgs,
      std::size_t... PIs,
      typename T = typename std::decay_t<E>::value_type>
      void initialise(xt::xexpression<E>& X, std::tuple<PosArgs...>&& posargs,
        std::index_sequence<PIs...>)
    {
      E& _X = X.derived_cast();

      POS f_pos(std::get<PIs>(std::move(posargs))...);
      f_pos(_X);
    }

    /**
     * @brief  method to evolve bird positions of the swarm
     *
     * @tparam E xtensor type for input and output vectors
     * @tparam F xtensor type for input and output of evaluation best
     * @tparam OBJ Functor type for objective function evaluation
     * @tparam POS Functor type for position evaluation
     * @tparam SEL Functor type for selection evaluation (Y best and X best)
     * @tparam MUT Functor type for mutation evaluation
     * @tparam PosArgs type of arguments for position evaluation functor
     * @tparam SelArgs type of arguments for selection evaluation functor
     * @tparam MutArgs type of arguments for mutation evaluation functor
     * @tparam PIs number of arguments for position evaluation functor
     * @tparam SIs number of arguments for selection evaluation functor
     * @tparam MIs number of arguments for mutation evaluation functor
     * @tparam T value type of xtensor
     * @param X vector with initial positions of the swarm
     * @param Xm1 vector with previous generation positions of the individuals comprising the swarm
     * @param YB vector with best evaluations of the individuals comprising the swarm
     * @param A vector with archived positions of the swarm individuals
     * @param objective_f functor for objective function evaluation
     * @param posargs tuple with arguments for position functor
     * @param selargs tuple with arguments for selection functor
     * @param mutargs tuple with arguments for velocity functor
     */
    template<class E, class F, class OBJ, class POS = Position_pso_ga, class SEL = Selection_best_pso_ga,
      class MUT = Mutation_functor_polynomial, typename... PosArgs, typename... SelArgs, typename... MutArgs,
      std::size_t... PIs, std::size_t... SIs, std::size_t... MIs,
      typename T = typename std::decay_t<E>::value_type>
      void evolve(xt::xexpression<E>& X, xt::xexpression<E>& Xm1, xt::xexpression<F>& YB,
        xt::xexpression<E>& A, OBJ objective_f, std::tuple<PosArgs...>&& posargs,
        std::tuple<SelArgs...>&& selargs, std::tuple<MutArgs...>&& mutargs,
        std::index_sequence<PIs...>, std::index_sequence<SIs...>, std::index_sequence<MIs...>)
    {

      POS pos_f(std::get<PIs>(std::move(posargs))...);
      SEL sel_f(std::get<SIs>(std::move(selargs))...);
      MUT mutation_f(std::get<MIs>(std::move(mutargs))...);

      E& position = X.derived_cast();
      E& position_m1 = Xm1.derived_cast();
      F& y_best = YB.derived_cast();
      E& archive = A.derived_cast();

      auto shape_of_population = position.shape();
      std::size_t individual_size = shape_of_population[0];
      std::size_t variable_size = shape_of_population[1];

      pos_f(position, position_m1, archive, y_best);

      F y = objective_f(position);

      sel_f(position, archive, y, y_best);

      mutation_f(position);

      position_m1 = position;

    }

    /**
     * @brief  method to evolve bird positions of the swarm
     *
     * @tparam E xtensor type for input and output vectors
     * @tparam F xtensor type for input and output of evaluation best
     * @tparam OBJ Functor type for objective function evaluation
     * @tparam POS Functor type for position evaluation
     * @tparam SEL Functor type for selection evaluation (Y best and X best)
     * @tparam MUT Functor type for mutation evaluation
     * @tparam TERM Functor type for termination
     * @tparam PosArgs type of arguments for position evaluation functor
     * @tparam SelArgs type of arguments for selection evaluation functor
     * @tparam MutArgs type of arguments for mutation evaluation functor
     * @tparam TermArgs type of arguments for termination functor
     * @tparam PIs number of arguments for position evaluation functor
     * @tparam SIs number of arguments for selection evaluation functor
     * @tparam MIs number of arguments for mutation evaluation functor
     * @tparam TIs number of arguments for termination functor
     * @tparam T value type of xtensor
     * @param X vector with initial positions of the swarm
     * @param Xm1 vector with positions of previous generation of the individuals comprising the swarm
     * @param YB vector with best evaluations of the individuals comprising the swarm
     * @param A vector with archived positions of the swarm individuals
     * @param objective_f functor for objective function evaluation
     * @param posargs tuple with arguments for position functor
     * @param selargs tuple with arguments for selection functor
     * @param mutargs tuple with arguments for mutation functor
     * @param termargs tuple with arguments for termination functor
     */
    template<class E, class F, class OBJ, class POS = Position_pso_ga, class SEL = Selection_best_pso_ga,
      class MUT = Mutation_functor_polynomial, class TERM = Terminate_gen_max,
      typename... PosArgs, typename... SelArgs, typename... MutArgs, typename... TermArgs,
      std::size_t... PIs, std::size_t... SIs, std::size_t... MIs, std::size_t... TIs,
      typename T = typename std::decay_t<E>::value_type>
      auto evolve(xt::xexpression<E>& X, xt::xexpression<E>& Xm1, xt::xexpression<F>& YB,
        xt::xexpression<E>& A, OBJ objective_f, std::tuple<PosArgs...>&& posargs,
        std::tuple<SelArgs...>&& selargs, std::tuple<MutArgs...>&& mutargs, std::tuple<TermArgs...>&& termargs,
        std::index_sequence<PIs...>, std::index_sequence<SIs...>, std::index_sequence<MIs...>, std::index_sequence<TIs...>)
    {

      POS pos_f(std::get<PIs>(std::move(posargs))...);
      SEL sel_f(std::get<SIs>(std::move(selargs))...);
      MUT mutation_f(std::get<MIs>(std::move(mutargs))...);
      TERM terminate_f(std::get<TIs>(std::move(termargs))...);

      E& position = X.derived_cast();
      E& position_m1 = XB.derived_cast();
      F& y_best = YB.derived_cast();
      E& archive = A.derived_cast();

      auto shape_of_population = position.shape();
      std::size_t individual_size = shape_of_population[0];
      std::size_t variable_size = shape_of_population[1];

      pos_f(position, position_m1, archive, y_best);

      F y = objective_f(position);

      sel_f(position, archive, y, y_best);

      mutation_f(position);

      position_m1 = position;

      return terminate_f(position, objective_f(position));
    }

  };
}

#endif
