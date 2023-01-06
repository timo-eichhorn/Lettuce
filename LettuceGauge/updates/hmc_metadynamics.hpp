#ifndef LETTUCE_HMC_METAD_HPP
#define LETTUCE_HMC_METAD_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../math/su3.hpp"
#include "../math/su3_exp.hpp"
#include "../actions/gauge/wilson_action.hpp"
#include "../observables/clover.hpp"
#include "../smearing/stout_smearing.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <complex>
//----------------------------------------
// Standard C headers
// ...

// We can reuse the integrators defined in hmc_gauge.hpp

namespace GaugeUpdates
{
    template<typename IntegratorT, typename ActionT>
    struct HMCMetaDKernel
    {
        private:
            GaugeField&                             U;
            GaugeField&                             U_copy;
            GaugeField&                             Momentum;
            IntegratorT&                            Integrator;
            ActionT&                                Action;
            std::uniform_real_distribution<floatT>& distribution_prob;
            // Metadynamics
            // double n_smear_meta;
            // double rho_stout_meta;
            MetaBiasPotential&                                     Metapotential;
            FullTensor                                             Clover;
            GaugeFieldSmeared                                      SmearedFields;
            GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants> Exp_consts;
            GaugeField                                             ForceFatLink;

            // The integrator needs to access the private member functions UpdateMomenta() and UpdateFields()
            friend IntegratorT;

            double MetaCharge() noexcept
            {
                SmearedFields[0] = U;
                for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
                {
                    StoutSmearing4D(SmearedFields[smear_count], SmearedFields[smear_count + 1], rho_stout);
                }
                return TopChargeGluonicSymm(SmearedFields[n_smear_meta]);
            }

            double MetaChargeWithConstants() noexcept
            {
                SmearedFields[0] = U;
                for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
                {
                    StoutSmearing4DWithConstants(SmearedFields[smear_count], SmearedFields[smear_count + 1], Exp_consts[smear_count], rho_stout);
                }
                // Calculate clover term and topological charge (we usually need the clover term later during the update, so better this way than directly calculating the charge)
                CalculateClover(SmearedFields[n_smear_meta], Clover);
                return TopChargeGluonicSymm(Clover);
            }

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
            // Reverse momenta for HMC reversibility test

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
            // Calculate topological force/fat-link contribution from the metapotential

            void CalculateTopologicalForce() noexcept
            {
                // This is the Metadynamics/fat-link contribution to the momenta
                // First we need to smear the fields n_smear_meta times and store all intermediate fields
                SmearedFields[0] = U;
                for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
                {
                    StoutSmearing4DWithConstants(SmearedFields[smear_count], SmearedFields[smear_count + 1], Exp_consts[smear_count], rho_stout);
                }
                // Now we need the derivative of the metapotential and the contribution of the clover term
                // Calculate clover term on field that was smeared the most
                CalculateClover(SmearedFields[n_smear_meta], Clover);
                // Calculate derivative of metapotential at CV_old
                // TODO: This includes the interpolation constant. Is this correct, or do we really need (V_i + V_{i + 1})/dQ (like in 1508.07270)?
                //       We could try to use a symmetric difference V(Q + 0.5 * dq) - V(Q - 0.5 * dq), but then we have to be careful with the edges...
                double CV_old {TopChargeGluonicSymm(Clover)};
                double potential_derivative {Metapotential.ReturnDerivative(CV_old)};
                // Calculate clover derivative
                #pragma omp parallel for
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int mu = 0; mu < 4; ++mu)
                {
                    site_coord current_site {t, x, y, z};
                    // TODO: This should be a negative sign, since the force is given by the negative derivative of the potential?
                    //       There is another minus later on in the momentum update
                    ForceFatLink(current_site, mu) = potential_derivative * CloverDerivative(SmearedFields[n_smear_meta], Clover, current_site, mu);
                }
                // std::cout << "Clover derivative:\n" << ForceFatLink({4,2,6,7,1}) << std::endl;
                // std::cout << "Momenta (Clover derivative) lie in algebra: " << SU3::Tests::Testsu3All(ForceFatLink, 1e-12) << std::endl;
                // Finally perform the stout force recursion
                // Exp is calculated inside the StoutForceRecrusion function, we only need to pass an array of fitting shape
                for (int smear_count = n_smear_meta; smear_count > 0; --smear_count)
                {
                    // TODO: Replace global variable rho_stout with parameter?
                    StoutForceRecursion(SmearedFields[smear_count - 1], SmearedFields[smear_count], ForceFatLink, Exp_consts[smear_count - 1], rho_stout);
                    // std::cout << ForceFatLink({4,2,6,7,1}) << std::endl;
                }
            }

            //-----
            // Update momenta for HMC

            void UpdateMomenta(const floatT epsilon) noexcept
            {
                CalculateTopologicalForce();
                #pragma omp parallel for
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int mu = 0; mu < 4; ++mu)
                {
                    link_coord current_link {t, x, y, z, mu};
                    Matrix_3x3 tmp {Action.Staple(U, current_link) * U(current_link).adjoint()};
                    Momentum(current_link) -= epsilon * i<floatT> * (beta / static_cast<floatT>(6.0) * SU3::Projection::Algebra(tmp) + ForceFatLink(current_link));
                }
                // std::cout << "Momenta lie in algebra: " << SU3::Tests::Testsu3All(Momentum, 1e-12) << std::endl;
            }

            //-----
            // Update gauge fields for HMC

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
                    U({t, x, y, z, mu}) = tmp_mat * U({t, x, y, z, mu});
                    SU3::Projection::GramSchmidt(U({t, x, y, z, mu}));
                }
                // std::cout << "new test: " << TestSU3All(U, 1e-8) << std::endl;
                // std::cout << "new test: " << U[1][3][4][7][2].determinant() << "\n" << U[1][3][4][7][2] * U[1][3][4][7][2].adjoint() << "\n" << std::endl;
                // std::cout << "Fields lie in group: " << SU3::TestSU3All(U, 1e-12) << std::endl;
            }

            [[nodiscard]]
            double Hamiltonian() const noexcept
            {
                double potential_energy {Action.Action(U)};
                double kinetic_energy   {0.0};
                // TODO: Momentum * Momentum.adjoint() or Momentum^2? Also is there a prefactor 0.5 or not?
                #pragma omp parallel for reduction(+: kinetic_energy)
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
            explicit HMCMetaDKernel(GaugeField& U_in, GaugeField& U_copy_in, GaugeField& Momentum_in, MetaBiasPotential& Metapotential_in, IntegratorT& Integrator_in, ActionT& Action_in, const int n_smear_meta , std::uniform_real_distribution<floatT>& distribution_prob_in) noexcept :
            U(U_in), U_copy(U_copy_in), Momentum(Momentum_in), Metapotential(Metapotential_in), Integrator(Integrator_in), Action(Action_in), distribution_prob(distribution_prob_in), SmearedFields(n_smear_meta + 1), Exp_consts(n_smear_meta)
            {}


            bool operator()(const int n_step, const bool metropolis_step) noexcept
            {
                // Copy old field so we can restore it in case the update gets rejected
                U_copy = U;
                // Generate random momenta and calculate energy before time evolution
                RandomMomentum();
                double CV_old     {MetaCharge()};
                double energy_old {Hamiltonian() + Metapotential.ReturnPotential(CV_old)};
                // Perform integration with chosen integrator
                Integrator(*this, n_step);
                //-----
                // Calculate energy after time evolution
                double CV_new     {MetaCharge()};
                double energy_new {Hamiltonian() + Metapotential.ReturnPotential(CV_new)};
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
                        Metapotential.SetCV_current(CV_new);
                        if constexpr(metapotential_updated)
                        {
                            Metapotential.UpdatePotential(CV_new);
                        }
                        acceptance_count_hmc += 1;
                        return true;
                    }
                    else
                    {
                        Metapotential.SetCV_current(CV_old);
                        U = U_copy;
                        return false;
                    }
                }
                else
                {
                    Metapotential.SetCV_current(CV_new);
                    if constexpr(metapotential_updated)
                    {
                        Metapotential.UpdatePotential(CV_new);
                    }
                    datalog << "DeltaH: " << DeltaH << std::endl;
                    return true;
                }
            }
    };
} // namespace GaugeUpdates


#endif // LETTUCE_HMC_METAD_HPP
