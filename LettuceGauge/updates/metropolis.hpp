#ifndef LETTUCE_METROPOLIS_HPP
#define LETTUCE_METROPOLIS_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../math/su3.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <random>
//----------------------------------------
// Standard C headers
#include <cmath>

//+---------------------------------------------------------------------------------+
//| This file provides a functor implementing a (multi-hit) Metropolis update for   |
//| SU(3) gauge theory, which works by multiplying the old link with a random matrix|
//| obtained by taking a step with random size in a random direction in the algebra |
//| and exponentiating the result. The range of the stepsize depends on the distri- |
//| bution 'distribution_unitary'. Generally, acceptance rates at or below 0.5 seem |
//| to be most efficient.                                                           |
//+---------------------------------------------------------------------------------+

template<typename ActionT, typename prngT>
struct MetropolisKernel
{
    private:
        GaugeField& U;
        // TODO: Check if the stencil_radius of the Action is larger than 1 to prevent incorrect masking/parallelization
        ActionT&    Action;
        prngT&      prng;
        int         n_hit;
        floatT      epsilon;
        // TODO: Should we track the acceptance rates in the functors? Might be annoying to deal with when changing parameters/creating new instances of functors...
        //       Also unclear how to combine with parallelization
    public:
        explicit MetropolisKernel(GaugeField& U_in, ActionT& Action_in, prngT& prng_in, const int n_hit_in, const floatT epsilon_in) noexcept :
        U(U_in), Action(Action_in), prng(prng_in), n_hit(n_hit_in), epsilon(epsilon_in)
        {}

        int operator()(const link_coord& current_link) const noexcept
        {
            Matrix_3x3 st           {Action.Staple(U, current_link)};
            Matrix_SU3 old_link     {U(current_link)};
            double     S_old        {Action.Local(old_link, st)};
            int        accept_count {0};

            // Perform multiple hits on the same link
            for (int n_hit = 0; n_hit < multi_hit; ++n_hit)
            {
                // auto start_multihit = std::chrono::high_resolution_clock::now();
                int        choice   {prng.UniformInt(current_link)};
                // Uniform real returns an integer in the interval [0, 1), so adjust accordingly 
                // TODO: Should this be a feature of the prng class?
                floatT     phi      {prng.UniformReal(current_link) * 2.0 * epsilon - epsilon};
                Matrix_SU3 new_link {old_link * SU3::RandomMatParallel(choice, phi)};
                // auto end_multihit = std::chrono::high_resolution_clock::now();
                // multihit_time += end_multihit - start_multihit;

                // auto start_accept_reject = std::chrono::high_resolution_clock::now();
                double     S_new    {Action.Local(new_link, st)};
                double     p        {std::exp(-S_new + S_old)};
                double     q        {prng.UniformReal()};

                // Ugly hack to avoid branches in parallel region
                // CAUTION: We would want to check if q <= p, since for beta = 0 everything should be accepted
                // Unfortunately signbit(0) returns false... Is there way to fix this?
                // bool accept {std::signbit(q - p)};
                // U[t][x][y][z][mu] = accept * new_link + (!accept) * old_link;
                // old_link = accept * new_link + (!accept) * old_link;
                // s = accept * sprime + (!accept) * s;
                // accept_count += accept;
                if (q <= p)
                {
                    U(current_link) = new_link;
                    old_link = new_link;
                    S_old = S_new;
                    accept_count += 1;
                }
                // auto end_accept_reject = std::chrono::high_resolution_clock::now();
                // accept_reject_time += end_accept_reject - start_accept_reject;
            }
            SU3::Projection::GramSchmidt(U(current_link));
            // TODO: Since we currently count how many of the individual hits are accepted, accept_count can generally exceed 1 and thus can't be a bool
            //       This is inconsistent compared to the other update functors. Is this okay, or should we rather only track if at least one out of the
            //       n_hit hits is accepted?
            return accept_count;
        }

        void SetNHit(const int n_hit_in) noexcept
        {
            n_hit = n_hit_in;
        }

        int GetNHit() const noexcept
        {
            return n_hit;
        }

        // TODO: This still relies on several global variables like metro_norm and metro_target_acceptance
        void AdjustEpsilon(const uint_fast64_t& acceptance_count) noexcept
        {
            epsilon = epsilon + (acceptance_count * metro_norm - static_cast<floatT>(metro_target_acceptance)) * static_cast<floatT>(0.2);
        }

        floatT GetEpsilon() const noexcept
        {
            return epsilon;
        }
};

#endif // LETTUCE_METROPOLIS_HPP
