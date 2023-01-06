#ifndef LETTUCE_METROPOLIS_HPP
#define LETTUCE_METROPOLIS_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../math/su3.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
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

// template<typename floatT>
template<typename ActionT>
struct MetropolisKernel
{
    private:
        GaugeField&                             Gluon;
        // TODO: Check if the stencil_radius of the Action is larger than 1 to prevent incorrect masking/parallelization
        ActionT&                                Action;
        int                                     n_hit;
        // TODO: These distributions should be thread-safe, so we can probably get away with only having one instance of each distribution?
        std::uniform_real_distribution<floatT>& distribution_prob;
        std::uniform_real_distribution<floatT>& distribution_unitary;
        std::uniform_int_distribution<int>&     distribution_choice;
        // TODO: Should we track the acceptance rates in the functors? Might be annoying to deal with when changing parameters/creating new instances of functors...
        //       Also unclear how to combine with parallelization
    public:
        explicit MetropolisKernel(GaugeField& Gluon_in, ActionT& Action_in, const int n_hit_in, std::uniform_real_distribution<floatT>& distribution_prob_in, std::uniform_real_distribution<floatT>& distribution_unitary_in, std::uniform_int_distribution<int>& distribution_choice_in) noexcept :
        Gluon(Gluon_in), Action(Action_in), n_hit(n_hit_in), distribution_prob(distribution_prob_in), distribution_unitary(distribution_unitary_in), distribution_choice(distribution_choice_in)
        {}

        int operator()(const link_coord& current_link) const noexcept
        {
            Matrix_3x3 st           {Action.Staple(Gluon, current_link)};
            Matrix_SU3 old_link     {Gluon(current_link)};
            double     S_old        {Action.Local(old_link, st)};
            int        accept_count {0};

            // Perform multiple hits on the same link
            for (int n_hit = 0; n_hit < multi_hit; ++n_hit)
            {
                #if defined(_OPENMP)
                int        choice       {distribution_choice(prng_vector[omp_get_thread_num()])};
                floatT     phi          {distribution_unitary(prng_vector[omp_get_thread_num()])};
                Matrix_SU3 new_link     {old_link * SU3::RandomMatParallel(choice, phi)};
                #else
                // auto start_multihit = std::chrono::high_resolution_clock::now();
                int        choice       {distribution_choice(generator_rand)};
                floatT     phi          {distribution_unitary(generator_rand)};
                Matrix_SU3 new_link     {old_link * SU3::RandomMatParallel(choice, phi)};
                // auto end_multihit = std::chrono::high_resolution_clock::now();
                // multihit_time += end_multihit - start_multihit;
                #endif

                // auto start_accept_reject = std::chrono::high_resolution_clock::now();
                double     S_new        {Action.Local(new_link, st)};
                double     p            {std::exp(-S_new + S_old)};
                // TODO: Does this help in any way? Also try out for Orelax
                // double p {std::exp(SLocalDiff(old_link - new_link, st))};
                #if defined(_OPENMP)
                double      q           {distribution_prob(prng_vector[omp_get_thread_num()])};
                #else
                double      q           {distribution_prob(generator_rand)};
                #endif

                // Ugly hack to avoid branches in parallel region
                // CAUTION: We would want to check if q <= p, since for beta = 0 everything should be accepted
                // Unfortunately signbit(0) returns false... Is there way to fix this?
                // bool accept {std::signbit(q - p)};
                // Gluon[t][x][y][z][mu] = accept * new_link + (!accept) * old_link;
                // old_link = accept * new_link + (!accept) * old_link;
                // s = accept * sprime + (!accept) * s;
                // accept_count += accept;
                if (q <= p)
                {
                    Gluon(current_link) = new_link;
                    old_link = new_link;
                    S_old = S_new;
                    accept_count += 1;
                }
                // auto end_accept_reject = std::chrono::high_resolution_clock::now();
                // accept_reject_time += end_accept_reject - start_accept_reject;
            }
            SU3::Projection::GramSchmidt(Gluon(current_link));
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
        floatT AdjustedEpsilon(const floatT epsilon, const uint_fast64_t& acceptance_count) const noexcept
        {
            return epsilon + (acceptance_count * metro_norm - static_cast<floatT>(metro_target_acceptance)) * static_cast<floatT>(0.2);
        }
};

//-----
// Metropolis update routine (original version)

// void MetropolisUpdate(GaugeField& Gluon, const int n_sweep, uint_fast64_t& acceptance_count, floatT& epsilon, std::uniform_real_distribution<floatT>& distribution_prob, std::uniform_int_distribution<int>& distribution_choice, std::uniform_real_distribution<floatT>& distribution_unitary)
// {
//     acceptance_count = 0;

//     // std::chrono::duration<double> staple_time {0.0};
//     // std::chrono::duration<double> local_time {0.0};
//     // std::chrono::duration<double> multihit_time {0.0};
//     // std::chrono::duration<double> accept_reject_time {0.0};
//     for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
//     for (int mu = 0; mu < 4; ++mu)
//     for (int eo = 0; eo < 2; ++eo)
//     {
//         // #pragma omp parallel for reduction(+: acceptance_count) shared(prng_vector) private(st, old_link, new_link, s, sprime) firstprivate(eo, mu)
//         #pragma omp parallel for reduction(+: acceptance_count) shared(prng_vector)
//         for (int t = 0; t < Nt; ++t)
//         for (int x = 0; x < Nx; ++x)
//         for (int y = 0; y < Ny; ++y)
//         {
//             int offset {((t + x + y) & 1) ^ eo};
//             for (int z = offset; z < Nz; z+=2)
//             {
//                 // auto start_staple = std::chrono::high_resolution_clock::now();
//                 Matrix_3x3 st {WilsonAction::Staple(Gluon, {t, x, y, z}, mu)};
//                 // auto end_staple = std::chrono::high_resolution_clock::now();
//                 // staple_time += end_staple - start_staple;

//                 // auto start_local = std::chrono::high_resolution_clock::now();
//                 Matrix_SU3 old_link {Gluon({t, x, y, z, mu})};
//                 double s {WilsonAction::Local(old_link, st)};
//                 // auto end_local = std::chrono::high_resolution_clock::now();
//                 // local_time += end_local - start_local;
//                 // std::array<int, multi_hit>    prng_choice_vec;
//                 // std::array<floatT, multi_hit> prng_unitary_vec;
//                 // std::array<floatT, multi_hit> prng_prob_vec;
//                 // for (int n_hit = 0; n_hit < multi_hit; ++n_hit)
//                 // {
//                 //     prng_choice_vec[n_hit] = distribution_choice(prng_vector[omp_get_thread_num()]);
//                 //     prng_unitary_vec[n_hit] = distribution_unitary(prng_vector[omp_get_thread_num()]);
//                 //     prng_prob_vec[n_hit] = distribution_prob(prng_vector[omp_get_thread_num()]);
//                 // }
//                 for (int n_hit = 0; n_hit < multi_hit; ++n_hit)
//                 {
//                     #if defined(_OPENMP)
//                     // int choice = prng_choice_vec[n_hit];
//                     // floatT phi = prng_unitary_vec[n_hit];
//                     int choice {distribution_choice(prng_vector[omp_get_thread_num()])};
//                     floatT phi {distribution_unitary(prng_vector[omp_get_thread_num()])};
//                     Matrix_SU3 new_link {old_link * SU3::RandomMatParallel(choice, phi)};
//                     #else
//                     // auto start_multihit = std::chrono::high_resolution_clock::now();
//                     int choice {distribution_choice(generator_rand)};
//                     floatT phi {distribution_unitary(generator_rand)};
//                     Matrix_SU3 new_link {old_link * SU3::RandomMatParallel(choice, phi)};
//                     // auto end_multihit = std::chrono::high_resolution_clock::now();
//                     // multihit_time += end_multihit - start_multihit;
//                     #endif

//                     // auto start_accept_reject = std::chrono::high_resolution_clock::now();
//                     double sprime {WilsonAction::Local(new_link, st)};
//                     double p {std::exp(-sprime + s)};
//                     // TODO: Does this help in any way? Also try out for Orelax
//                     // double p {std::exp(SLocalDiff(old_link - new_link, st))};
//                     #if defined(_OPENMP)
//                     double q {distribution_prob(prng_vector[omp_get_thread_num()])};
//                     // double q = prng_prob_vec[n_hit];
//                     #else
//                     double q {distribution_prob(generator_rand)};
//                     #endif

//                     // Ugly hack to avoid branches in parallel region
//                     // CAUTION: We would want to check if q <= p, since for beta = 0 everything should be accepted
//                     // Unfortunately signbit(0) returns false... Is there way to fix this?
//                     // bool accept {std::signbit(q - p)};
//                     // Gluon[t][x][y][z][mu] = accept * new_link + (!accept) * old_link;
//                     // old_link = accept * new_link + (!accept) * old_link;
//                     // s = accept * sprime + (!accept) * s;
//                     // acceptance_count += accept;
//                     if (q <= p)
//                     {
//                         Gluon({t, x, y, z, mu}) = new_link;
//                         old_link = new_link;
//                         s = sprime;
//                         acceptance_count += 1;
//                     }
//                     // auto end_accept_reject = std::chrono::high_resolution_clock::now();
//                     // accept_reject_time += end_accept_reject - start_accept_reject;
//                 }
//                 SU3::Projection::GramSchmidt(Gluon({t, x, y, z, mu}));
//             }
//         }
//     }
//     // TODO: Test which acceptance rate is best. Initially had 0.8 as target, but 0.5 seems to thermalize much faster!
//     // Adjust PRNG width to target mean acceptance rate of 0.5
//     epsilon += (acceptance_count * metro_norm - static_cast<floatT>(metro_target_acceptance)) * static_cast<floatT>(0.2);
//     // cout << "staple_time: " << staple_time.count() << "\n";
//     // cout << "local_time: " << local_time.count() << "\n";
//     // cout << "multihit_time: " << multihit_time.count() << "\n";
//     // cout << "accept_reject_time: " << accept_reject_time.count() << endl;
// }

#endif // LETTUCE_METROPOLIS_HPP
