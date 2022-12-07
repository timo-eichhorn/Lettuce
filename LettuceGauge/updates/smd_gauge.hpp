#ifndef LETTUCE_SMD_GAUGE_HPP
#define LETTUCE_SMD_GAUGE_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../math/su3.hpp"
#include "../math/su3_exp.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <complex>
//----------------------------------------
// Standard C headers
// ...

// TODO: Implement!
// In contrast to the HMC, you need to make sure that the Momentum field is not reused/overwritten by other functions,
// since the momenta are only partially refreshed at the start of a new trajectory!
namespace GaugeUpdates
{
    template<typename IntegratorT, typename ActionT>
    struct SMDKernel
    {
        private:
            GaugeField&  Gluon;
            GaugeField&  Gluon_copy;
            GaugeField&  Momentum;
            IntegratorT& Integrator;
            ActionT&     Action;
            std::uniform_real_distribution<floatT>& distribution_prob;

            // The integrator needs to access the private member functions UpdateMomenta() and UpdateFields()
            friend IntegratorT;

            void RandomMomentum() const noexcept
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
            // Reverse momenta for SMD reversibility test

            void ReverseMomenta() const noexcept
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
            // Update momenta for SMD

            void UpdateMomenta(const floatT epsilon) const noexcept
            {
                #pragma omp parallel for
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int mu = 0; mu < 4; ++mu)
                {
                    link_coord current_link {t, x, y, z, mu};
                    Matrix_3x3 tmp {Action.Staple(Gluon, current_link) * Gluon(current_link).adjoint()};
                    Momentum(current_link) -= epsilon * i<floatT> * beta / 6.0 * SU3::Projection::Algebra(tmp);
                }
                // std::cout << "Momenta lie in algebra: " << SU3::Tests::Testsu3All(Momentum, 1e-12) << std::endl;
            }

            //-----
            // Update gauge fields for SMD

            void UpdateFields(const floatT epsilon) const noexcept
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
            double Hamiltonian() const noexcept
            {
                // double potential_energy {WilsonAction::Action(Gluon)};
                double potential_energy {Action.Action(Gluon)};
                double kinetic_energy   {0.0};
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
        public:
            explicit SMDKernel(GaugeField& Gluon_in, GaugeField& Gluon_copy_in, GaugeField& Momentum_in, IntegratorT& Integrator_in, ActionT& Action_in, std::uniform_real_distribution<floatT>& distribution_prob_in) noexcept :
            Gluon(Gluon_in), Gluon_copy(Gluon_copy_in), Momentum(Momentum_in), Integrator(Integrator_in), Action(Action_in), distribution_prob(distribution_prob_in)
            {}


            bool operator()(const int n_step, const bool metropolis_step) const noexcept
            {
                // Copy old field so we can restore it in case the update gets rejected
                Gluon_copy = Gluon;
                // Generate random momenta and calculate energy before time evolution
                RandomMomentum();
                double energy_old {Hamiltonian()};
                // Perform integration with chosen integrator
                Integrator(*this, n_step);
                //-----
                // Calculate energy after time evolution
                double energy_new {Hamiltonian()};
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
                        acceptance_count_smd += 1;
                        return true;
                    }
                    else
                    {
                        Gluon = Gluon_copy;
                        ReverseMomenta();
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
} // namespace GaugeUpdates

#endif // LETTUCE_SMD_GAUGE_HPP
