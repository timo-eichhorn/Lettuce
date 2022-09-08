#ifndef LETTUCE_METROPOLIS_HPP
#define LETTUCE_METROPOLIS_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
// ...
//----------------------------------------
// Standard C headers
// ...

//+---------------------------------------------------------------------------------+
//| This file provides a functor implementing a (multi-hit) Metropolis update for   |
//| SU(3) gauge theory, which works by multiplying the old link with a random matrix|
//| obtained by taking a step with random size in a random direction in the algebra |
//| and exponentiating the result. The range of the stepsize depends on the distri- |
//| bution 'distribution_unitary'. Generally, acceptance rates at or below 0.5 seem |
//| to be most efficient.                                                           |
//+---------------------------------------------------------------------------------+

// template<typename floatT>
struct MetropolisKernel
{
    private:
        // TODO: Add action as parameter, so we can update with respect to different actions
        GaugeField&                             Gluon;
        int                                     n_hit;
        // TODO: These distributions should be thread-safe, so we can probably get away with only having one instance of each distribution?
        std::uniform_real_distribution<floatT>& distribution_prob;
        std::uniform_real_distribution<floatT>& distribution_unitary;
        std::uniform_int_distribution<int>&     distribution_choice;
        // TODO: Should we track the acceptance rates in the functors? Might be annoying to deal with when changing parameters/creating new instances of functors...
        //       Also unclear how to combine with parallelization
    public:
        explicit MetropolisKernel(GaugeField& Gluon_in, const int n_hit_in, std::uniform_real_distribution<floatT>& distribution_prob_in, std::uniform_real_distribution<floatT>& distribution_unitary_in, std::uniform_int_distribution<int>& distribution_choice_in) noexcept :
        Gluon(Gluon_in), n_hit(n_hit_in), distribution_prob(distribution_prob_in), distribution_unitary(distribution_unitary_in), distribution_choice(distribution_choice_in)
        {}

        int operator()(const link_coord& current_link) const noexcept
        {
            Matrix_3x3 st       {WilsonAction::Staple(Gluon, current_link)};
            Matrix_SU3 old_link {Gluon(current_link)};
            double     S_old    {WilsonAction::Local(old_link, st)};

            // Perform multiple hits on the same link
            for (int n_hit = 0; n_hit < multi_hit; ++n_hit)
            {
                int        accept_count {0};
                #if defined(_OPENMP)
                int        choice       {distribution_choice(prng_vector[omp_get_thread_num()])};
                floatT     phi          {distribution_unitary(prng_vector[omp_get_thread_num()])};
                Matrix_SU3 new_link     {old_link * RandomSU3Parallel(choice, phi)};
                #else
                // auto start_multihit = std::chrono::high_resolution_clock::now();
                int        choice       {distribution_choice(generator_rand)};
                floatT     phi          {distribution_unitary(generator_rand)};
                Matrix_SU3 new_link     {old_link * RandomSU3Parallel(choice, phi)};
                // auto end_multihit = std::chrono::high_resolution_clock::now();
                // multihit_time += end_multihit - start_multihit;
                #endif

                // auto start_accept_reject = std::chrono::high_resolution_clock::now();
                double     S_new        {WilsonAction::Local(new_link, st)};
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
}

#endif // LETTUCE_METROPOLIS_HPP
