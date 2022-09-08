#ifndef LETTUCE_HMC_GAUGE_HPP
#define LETTUCE_HMC_GAUGE_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../math/su3.hpp"
#include "../math/su3_exp.hpp"
#include "../actions/gauge/wilson_action.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <complex>
//----------------------------------------
// Standard C headers
// ...

//-----
// Generate random momenta for HMC

namespace HMC
{
    void RandomMomentum(GaugeField& Momentum) noexcept
    {
        #pragma omp parallel for
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            // Generate 8 random numbers as basis coefficients
            floatT phi1 {ndist_vector[omp_get_thread_num()](prng_vector[omp_get_thread_num()])};
            floatT phi2 {ndist_vector[omp_get_thread_num()](prng_vector[omp_get_thread_num()])};
            floatT phi3 {ndist_vector[omp_get_thread_num()](prng_vector[omp_get_thread_num()])};
            floatT phi4 {ndist_vector[omp_get_thread_num()](prng_vector[omp_get_thread_num()])};
            floatT phi5 {ndist_vector[omp_get_thread_num()](prng_vector[omp_get_thread_num()])};
            floatT phi6 {ndist_vector[omp_get_thread_num()](prng_vector[omp_get_thread_num()])};
            floatT phi7 {ndist_vector[omp_get_thread_num()](prng_vector[omp_get_thread_num()])};
            floatT phi8 {ndist_vector[omp_get_thread_num()](prng_vector[omp_get_thread_num()])};
            // Random momentum in su(3) given by phi_i * T^i (where T^i is 0.5 * i-th Gell-Mann matrix)
            // Technically A is a traceless hermitian matrix, while su(3) matrices are anti-hermitian
            Matrix_3x3 A;
            A << std::complex<floatT>(phi3 + phi8/sqrt(3.0),0.0),std::complex<floatT>(phi1,-phi2),std::complex<floatT>(phi4,-phi5),
                 std::complex<floatT>(phi1,phi2),                std::complex<floatT>(-phi3 + phi8/sqrt(3.0),0.0),std::complex<floatT>(phi6,-phi7),
                 std::complex<floatT>(phi4,phi5),                std::complex<floatT>(phi6,phi7),std::complex<floatT>(-2.0*phi8/sqrt(3.0),0.0);
            Momentum({t, x, y, z, mu}) = static_cast<floatT>(0.5) * A;
        }
        // std::cout << "Random momenta lie in algebra: " << SU3::Tests::Testsu3All(Momentum, 1e-12) << std::endl;
    }

    //-----
    // Reverse momenta for HMC reversibility test

    void ReverseMomenta(GaugeField& Momentum) noexcept
    {
        #pragma omp parallel for
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            Momentum({t, x, y, z, mu}) = -Momentum({t, x, y, z, mu});
        }
        // std::cout << "Momenta lie in algebra: " << SU3::Tests::Testsu3All(Momentum, 1e-12) << std::endl;
    }

    //-----
    // Update momenta for HMC

    void UpdateMomenta(GaugeField& Gluon, GaugeField& Momentum, const floatT epsilon) noexcept
    {
        #pragma omp parallel for
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            link_coord current_link {t, x, y, z, mu};
            Matrix_3x3 st {WilsonAction::Staple(Gluon, current_link)};
            Matrix_3x3 tmp {st * Gluon(current_link).adjoint() - Gluon(current_link) * st.adjoint()};
            Momentum(current_link) -= epsilon * i<floatT> * beta / static_cast<floatT>(12.0) * (tmp - static_cast<floatT>(1.0/3.0) * tmp.trace() * Matrix_3x3::Identity());
        }
        // std::cout << "Momenta lie in algebra: " << SU3::Tests::Testsu3All(Momentum, 1e-12) << std::endl;
    }

    //-----
    // Update gauge fields for HMC

    void UpdateFields(GaugeField& Gluon, GaugeField& Momentum, const floatT epsilon) noexcept
    {
        #pragma omp parallel for
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            // Matrix_3x3 tmp_mat {(i<floatT> * epsilon * Momentum({t, x, y, z, mu})).exp()};
            Matrix_3x3 tmp_mat {SU3::exp(epsilon * Momentum({t, x, y, z, mu}))};
            Gluon({t, x, y, z, mu}) = tmp_mat * Gluon({t, x, y, z, mu});
            SU3::Projection::GramSchmidt(Gluon({t, x, y, z, mu}));
        }
        // std::cout << "new test: " << TestSU3All(Gluon, 1e-8) << std::endl;
        // std::cout << "new test: " << Gluon[1][3][4][7][2].determinant() << "\n" << Gluon[1][3][4][7][2] * Gluon[1][3][4][7][2].adjoint() << "\n" << std::endl;
        // std::cout << "Fields lie in group: " << SU3::TestSU3All(Gluon, 1e-12) << std::endl;
    }

    [[nodiscard]]
    double Hamiltonian(const GaugeField& Gluon, const GaugeField& Momentum) noexcept
    {
        double potential_energy {WilsonAction::Action(Gluon)};
        double kinetic_energy {0.0};
        // TODO: Momentum * Momentum.adjoint() or Momentum^2? Also is there a prefactor 0.5 or not?
        #pragma omp parallel for reduction(+:kinetic_energy)
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            kinetic_energy += std::real((Momentum({t, x, y, z, mu}) * Momentum({t, x, y, z, mu})).trace());
        }
        // std::cout << "kinetic_energy: " << kinetic_energy << " potential_energy: " << potential_energy << std::endl;
        return potential_energy + kinetic_energy;
    }

    // Leapfrog integrator for HMC
    void Leapfrog(GaugeField& Gluon, GaugeField& Momentum, const int n_step) noexcept
    {
        // Calculate stepsize epsilon from n_step
        floatT epsilon {static_cast<floatT>(1.0)/n_step};
        // Perform integration
        // Momentum updates are merged in the loop
        UpdateMomenta(Gluon, Momentum, 0.5 * epsilon);
        for (int step_count = 0; step_count < n_step - 1; ++step_count)
        {
            UpdateFields(Gluon, Momentum, epsilon);
            UpdateMomenta(Gluon, Momentum, epsilon);
        }
        UpdateFields(Gluon, Momentum, epsilon);
        UpdateMomenta(Gluon, Momentum, 0.5 * epsilon);
    }
    //-----
    // Omelyan-Mryglod-Folk second order minimum norm integrator (improved leapfrog)
    // cf. hep-lat/0505020
    // NOTE: This version doesn't use merged momentum updates and is slightly less efficient than the one below
    void OMF_2_slow(GaugeField& Gluon, GaugeField& Momentum, const int n_step) noexcept
    {
        // Calculate stepsize epsilon from n_step
        floatT epsilon {static_cast<floatT>(1.0)/n_step};
        // Integrator constants
        double alpha {0.1931833275037836 * epsilon};
        double beta  {0.5 * epsilon};
        double gamma {(1.0 - 2.0 * 0.1931833275037836) * epsilon};
        // Perform integration
        for (int step_count = 0; step_count < n_step; ++step_count)
        {
            UpdateMomenta(Gluon, Momentum, alpha);
            UpdateFields(Gluon, Momentum, beta);
            UpdateMomenta(Gluon, Momentum, gamma);
            UpdateFields(Gluon, Momentum, beta);
            UpdateMomenta(Gluon, Momentum, alpha);
        }
    }
    //-----
    // Omelyan-Mryglod-Folk second order minimum norm integrator (improved leapfrog)
    // cf. hep-lat/0505020
    void OMF_2(GaugeField& Gluon, GaugeField& Momentum, const int n_step) noexcept
    {
        // Calculate stepsize epsilon from n_step
        floatT epsilon {static_cast<floatT>(1.0)/n_step};
        // Integrator constants
        double alpha {0.1931833275037836 * epsilon};
        double beta  {0.5 * epsilon};
        double gamma {(1.0 - 2.0 * 0.1931833275037836) * epsilon};
        // Perform integration
        // Momentum updates are merged in the loop
        UpdateMomenta(Gluon, Momentum, alpha);
        UpdateFields(Gluon, Momentum, beta);
        UpdateMomenta(Gluon, Momentum, gamma);
        UpdateFields(Gluon, Momentum, beta);
        for (int step_count = 0; step_count < n_step - 1; ++step_count)
        {
            UpdateMomenta(Gluon, Momentum, 2.0 * alpha);
            UpdateFields(Gluon, Momentum, beta);
            UpdateMomenta(Gluon, Momentum, gamma);
            UpdateFields(Gluon, Momentum, beta);
        }
        UpdateMomenta(Gluon, Momentum, alpha);
    }
    //-----
    // Omelyan-Mryglod-Folk fourth order minimum norm integrator
    // cf. hep-lat/0505020
    // NOTE: This version doesn't use merged momentum updates and is slightly less efficient than the one below
    void OMF_4_slow(GaugeField& Gluon, GaugeField& Momentum, const int n_step) noexcept
    {
        // Calculate stepsize epsilon from n_step
        floatT epsilon {static_cast<floatT>(1.0)/n_step};
        // Integrator constants
        double alpha {0.08398315262876693 * epsilon};
        double beta  {0.2539785108410595 * epsilon};
        double gamma {0.6822365335719091 * epsilon};
        double delta {-0.03230286765269967 * epsilon};
        double mu    {(0.5 - 0.6822365335719091 - 0.08398315262876693) * epsilon};
        double nu    {(1.0 - 2.0 * -0.03230286765269967 - 2.0 * 0.2539785108410595) * epsilon};
        // Perform integration
        // Momentum updates are not merged in the loop
        for (int step_count = 0; step_count < n_step; ++step_count)
        {
            UpdateMomenta(Gluon, Momentum, alpha);
            UpdateFields(Gluon, Momentum, beta);
            UpdateMomenta(Gluon, Momentum, gamma);
            UpdateFields(Gluon, Momentum, delta);

            UpdateMomenta(Gluon, Momentum, mu);
            UpdateFields(Gluon, Momentum, nu);
            UpdateMomenta(Gluon, Momentum, mu);

            UpdateFields(Gluon, Momentum, delta);
            UpdateMomenta(Gluon, Momentum, gamma);
            UpdateFields(Gluon, Momentum, beta);
            UpdateMomenta(Gluon, Momentum, alpha);
        }
    }
    //-----
    // Omelyan-Mryglod-Folk fourth order minimum norm integrator
    // cf. hep-lat/0505020
    void OMF_4(GaugeField& Gluon, GaugeField& Momentum, const int n_step) noexcept
    {
        // Calculate stepsize epsilon from n_step
        floatT epsilon {static_cast<floatT>(1.0)/n_step};
        // Integrator constants
        double alpha {0.08398315262876693 * epsilon};
        double beta  {0.2539785108410595 * epsilon};
        double gamma {0.6822365335719091 * epsilon};
        double delta {-0.03230286765269967 * epsilon};
        double mu    {(0.5 - 0.6822365335719091 - 0.08398315262876693) * epsilon};
        double nu    {(1.0 - 2.0 * -0.03230286765269967 - 2.0 * 0.2539785108410595) * epsilon};
        // Perform integration
        // Momentum updates are merged in the loop
        UpdateMomenta(Gluon, Momentum, alpha);
        UpdateFields(Gluon, Momentum, beta);
        UpdateMomenta(Gluon, Momentum, gamma);
        UpdateFields(Gluon, Momentum, delta);

        UpdateMomenta(Gluon, Momentum, mu);
        UpdateFields(Gluon, Momentum, nu);
        UpdateMomenta(Gluon, Momentum, mu);

        UpdateFields(Gluon, Momentum, delta);
        UpdateMomenta(Gluon, Momentum, gamma);
        UpdateFields(Gluon, Momentum, beta);
        for (int step_count = 0; step_count < n_step - 1; ++step_count)
        {
            UpdateMomenta(Gluon, Momentum, 2.0 * alpha);
            UpdateFields(Gluon, Momentum, beta);
            UpdateMomenta(Gluon, Momentum, gamma);
            UpdateFields(Gluon, Momentum, delta);

            UpdateMomenta(Gluon, Momentum, mu);
            UpdateFields(Gluon, Momentum, nu);
            UpdateMomenta(Gluon, Momentum, mu);

            UpdateFields(Gluon, Momentum, delta);
            UpdateMomenta(Gluon, Momentum, gamma);
            UpdateFields(Gluon, Momentum, beta);
        }
        UpdateMomenta(Gluon, Momentum, alpha);
    }

    //-----
    // HMC for pure gauge theory
    // TODO: Constraint on FuncT?
    // TODO: Add tau parameter (trajectory time) and pass to integrator functions

    template<typename FuncT>
    bool HMCGauge(GaugeField& Gluon, GaugeField& Gluon_copy, GaugeField& Momentum, uint_fast64_t& acceptance_count_hmc, FuncT&& Integrator, const int n_step, const bool metropolis_step, std::uniform_real_distribution<floatT>& distribution_prob) noexcept
    {
        // Copy old field so we can restore it in case the update gets rejected
        Gluon_copy = Gluon;
        // Generate random momenta and calculate energy before time evolution
        RandomMomentum(Momentum);
        double energy_old {Hamiltonian(Gluon, Momentum)};
        // Perform integration with chosen integrator
        Integrator(Gluon, Momentum, n_step);
        //-----
        // Reversibility test
        // ReverseMomenta(Momentum);
        // UpdateMomenta(Gluon, Momentum, 0.5 * epsilon);
        // for (int step_count = 0; step_count < n_step - 1; ++step_count)
        // {
        //     UpdateFields(Gluon, Momentum, epsilon);
        //     UpdateMomenta(Gluon, Momentum, epsilon);
        // }
        // UpdateFields(Gluon, Momentum, epsilon);
        // UpdateMomenta(Gluon, Momentum, 0.5 * epsilon);
        //-----
        // Calculate energy after time evolution
        double energy_new {Hamiltonian(Gluon, Momentum)};
        // Metropolis accept-reject step
        double p {std::exp(-energy_new + energy_old)};
        #if defined(_OPENMP)
        double q {distribution_prob(prng_vector[omp_get_thread_num()])};
        #else
        double q {distribution_prob(generator_rand)};
        #endif
        // TODO: Probably shouldnt use a global variable for DeltaH?
        DeltaH = energy_new - energy_old;
        if (metropolis_step)
        {
            // datalog << "DeltaH: " << DeltaH << std::endl;
            if (q <= p)
            {
                acceptance_count_hmc += 1;
                return true;
            }
            else
            {
                Gluon = Gluon_copy;
                return false;
            }
        }
        else
        {
            datalog << "DeltaH: " << DeltaH << std::endl;
            return true;
        }
    }

    //-----
    // WIP: HMC functor implementation

    template<typename FuncT>
    struct HMCKernel
    {
        private:
            GaugeField& Gluon;
            GaugeField& Gluon_copy;
            GaugeField& Momentum;
            FuncT&      Integrator;
            std::uniform_real_distribution<floatT>& distribution_prob;
            // int n_step;
            // bool metropolis_step;
        public:
            explicit HMCKernel(GaugeField& Gluon_in, GaugeField& Gluon_copy_in, GaugeField& Momentum_in, FuncT& Integrator_in, std::uniform_real_distribution<floatT>& distribution_prob_in) noexcept :
            Gluon(Gluon_in), Gluon_copy(Gluon_copy_in), Momentum(Momentum_in), Integrator(Integrator_in), distribution_prob(distribution_prob_in)
            {}


            bool operator()(const int n_step, const bool metropolis_step) const noexcept
            {
                // Copy old field so we can restore it in case the update gets rejected
                Gluon_copy = Gluon;
                // Generate random momenta and calculate energy before time evolution
                RandomMomentum(Momentum);
                double energy_old {Hamiltonian(Gluon, Momentum)};
                // Perform integration with chosen integrator
                Integrator(Gluon, Momentum, n_step);
                //-----
                // Reversibility test
                // ReverseMomenta(Momentum);
                // UpdateMomenta(Gluon, Momentum, 0.5 * epsilon);
                // for (int step_count = 0; step_count < n_step - 1; ++step_count)
                // {
                //     UpdateFields(Gluon, Momentum, epsilon);
                //     UpdateMomenta(Gluon, Momentum, epsilon);
                // }
                // UpdateFields(Gluon, Momentum, epsilon);
                // UpdateMomenta(Gluon, Momentum, 0.5 * epsilon);
                //-----
                // Calculate energy after time evolution
                double energy_new {Hamiltonian(Gluon, Momentum)};
                // Metropolis accept-reject step
                double p {std::exp(-energy_new + energy_old)};
                #if defined(_OPENMP)
                double q {distribution_prob(prng_vector[omp_get_thread_num()])};
                #else
                double q {distribution_prob(generator_rand)};
                #endif
                // TODO: Probably shouldnt use a global variable for DeltaH?
                DeltaH = energy_new - energy_old;
                if (metropolis_step)
                {
                    // datalog << "DeltaH: " << DeltaH << std::endl;
                    if (q <= p)
                    {
                        acceptance_count_hmc += 1;
                        return true;
                    }
                    else
                    {
                        Gluon = Gluon_copy;
                        return false;
                    }
                }
                else
                {
                    datalog << "DeltaH: " << DeltaH << std::endl;
                    return true;
                }
            }
    };
} // namespace HMC

#endif // LETTUCE_HMC_GAUGE_HPP
