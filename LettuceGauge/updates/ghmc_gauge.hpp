#ifndef LETTUCE_GHMC_GAUGE_HPP
#define LETTUCE_GHMC_GAUGE_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../coords.hpp"
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

// TODO: Condense (parts of) HMC and generalized HMC to common molecular dynamcis kernel?
// In contrast to the HMC, you need to make sure that the Momentum field is not reused/overwritten by other functions,
// since the momenta are only partially refreshed at the start of a new trajectory!
namespace GaugeUpdates
{
    template<typename IntegratorT, typename ActionT, typename prngT>
    struct GeneralizedHMCKernel
    {
        private:
            GaugeField&  U;
            GaugeField&  U_copy;
            // Warning: Since the momenta from the previous trajectory contribute to the new momenta, you need to make sure not to change the momenta from the outside!
            // TODO: Change this so the GeneralizedHMCKernel owns the momentum field? Tradeoff between safety and performance (usually functors should be cheap to construct)
            GaugeField&  Momentum;
            GaugeField&  Momentum_copy;
            IntegratorT& Integrator;
            ActionT&     Action;
            prngT&       prng;
        public:
            // Mixing parameter for old momenta and new Gaussian momenta, where:
            //     p_new = cos(mixing_angle) * p_old + sin(mixing_angle) * p_random
            // Evidently, the original HMC is recovered for mixing_angle = pi/2
            double       mixing_angle;
            double       trajectory_length;
        private:

            // The integrator needs to access the private member functions UpdateMomenta() and UpdateFields()
            friend IntegratorT;

            // void RandomMomentum(GaugeField& P) const noexcept
            // {
            //     #pragma omp parallel for collapse(2)
            //     for (int t = 0; t < Nt; ++t)
            //     for (int x = 0; x < Nx; ++x)
            //     for (int y = 0; y < Ny; ++y)
            //     for (int z = 0; z < Nz; ++z)
            //     for (int mu = 0; mu < 4; ++mu)
            //     {
            //         // Generate 8 random numbers as basis coefficients
            //         link_coord current_link {t, x, y, z, mu};
            //         floatT     phi1         {prng.Gaussian(current_link)};
            //         floatT     phi2         {prng.Gaussian(current_link)};
            //         floatT     phi3         {prng.Gaussian(current_link)};
            //         floatT     phi4         {prng.Gaussian(current_link)};
            //         floatT     phi5         {prng.Gaussian(current_link)};
            //         floatT     phi6         {prng.Gaussian(current_link)};
            //         floatT     phi7         {prng.Gaussian(current_link)};
            //         floatT     phi8         {prng.Gaussian(current_link)};

            //         // Random momentum in su(3) given by i * phi_i * T^i (where T^i is 0.5 * i-th Gell-Mann/traceless hermitian matrix)
            //         Matrix_3x3 A;
            //         A << std::complex<floatT>(0.0,phi3 + phi8/sqrt(3.0)), std::complex<floatT>(phi2,phi1),                  std::complex<floatT>(phi5,phi4),
            //              std::complex<floatT>(-phi2,phi1),                std::complex<floatT>(0.0,-phi3 + phi8/sqrt(3.0)), std::complex<floatT>(phi7,phi6),
            //              std::complex<floatT>(-phi5,phi4),                std::complex<floatT>(-phi7,phi6),                 std::complex<floatT>(0.0,-2.0*phi8/sqrt(3.0));
            //         P(current_link) = static_cast<floatT>(0.5) * A;
            //     }
            //     // std::cout << "Random momenta lie in algebra: " << SU3::Tests::Testsu3All(Momentum, 1e-12) << std::endl;
            // }

            // Partially refresh momenta
            void RotateMomentum(GaugeField& P) const noexcept
            {
                double old_momentum_contribution    {std::cos(mixing_angle)};
                double random_momentum_contribution {std::sin(mixing_angle)};
                #pragma omp parallel for collapse(2)
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int mu = 0; mu < 4; ++mu)
                {
                    // Generate 8 random numbers as basis coefficients
                    link_coord current_link {t, x, y, z, mu};
                    floatT     phi1         {prng.Gaussian(current_link)};
                    floatT     phi2         {prng.Gaussian(current_link)};
                    floatT     phi3         {prng.Gaussian(current_link)};
                    floatT     phi4         {prng.Gaussian(current_link)};
                    floatT     phi5         {prng.Gaussian(current_link)};
                    floatT     phi6         {prng.Gaussian(current_link)};
                    floatT     phi7         {prng.Gaussian(current_link)};
                    floatT     phi8         {prng.Gaussian(current_link)};

                    // Random momentum in su(3) given by i * phi_i * T^i (where T^i is 0.5 * i-th Gell-Mann/traceless hermitian matrix)
                    Matrix_3x3 A;
                    A << std::complex<floatT>(0.0,phi3 + phi8/sqrt(3.0)), std::complex<floatT>(phi2,phi1),                  std::complex<floatT>(phi5,phi4),
                         std::complex<floatT>(-phi2,phi1),                std::complex<floatT>(0.0,-phi3 + phi8/sqrt(3.0)), std::complex<floatT>(phi7,phi6),
                         std::complex<floatT>(-phi5,phi4),                std::complex<floatT>(-phi7,phi6),                 std::complex<floatT>(0.0,-2.0*phi8/sqrt(3.0));
                    P(current_link) = old_momentum_contribution * P(current_link) + random_momentum_contribution * static_cast<floatT>(0.5) * A;
                }
                // std::cout << "Random momenta lie in algebra: " << SU3::Tests::Testsu3All(Momentum, 1e-12) << std::endl;
            }

            //-----
            // Reverse momenta

            void ReverseMomenta(GaugeField& P) const noexcept
            {
                #pragma omp parallel for collapse(2)
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int mu = 0; mu < 4; ++mu)
                {
                    P({t, x, y, z, mu}) = -P({t, x, y, z, mu});
                }
                // std::cout << "Momenta lie in algebra: " << SU3::Tests::Testsu3All(Momentum, 1e-12) << std::endl;
            }

            //-----
            // Update momenta for generalized HMC

            void UpdateMomenta(const floatT epsilon) const noexcept
            {
                #pragma omp parallel for collapse(2)
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int mu = 0; mu < 4; ++mu)
                {
                    link_coord current_link {t, x, y, z, mu};
                    Matrix_3x3 tmp {Action.Staple(U, current_link) * U(current_link).adjoint()};
                    Momentum(current_link) += epsilon * beta / 6.0 * SU3::Projection::Algebra(tmp);
                }
                // std::cout << "Momenta lie in algebra: " << SU3::Tests::Testsu3All(Momentum, 1e-12) << std::endl;
            }

            //-----
            // Update gauge fields for generalized HMC

            void UpdateFields(const floatT epsilon) const noexcept
            {
                #pragma omp parallel for collapse(2)
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int mu = 0; mu < 4; ++mu)
                {
                    // Matrix_3x3 tmp_mat {(i<floatT> * epsilon * Momentum({t, x, y, z, mu})).exp()};
                    // Multiplication with -i since SU3::exp() expects a traceless hermitian 3x3 matrix Mat and returns exp(i * Mat)
                    Matrix_3x3 tmp_mat {SU3::exp(-i<floatT> * epsilon * Momentum({t, x, y, z, mu}))};
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
                #pragma omp parallel for collapse(2) reduction(+: kinetic_energy)
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int mu = 0; mu < 4; ++mu)
                {
                    kinetic_energy -= std::real((Momentum({t, x, y, z, mu}) * Momentum({t, x, y, z, mu})).trace());
                }
                // std::cout << "kinetic_energy: " << kinetic_energy << " potential_energy: " << potential_energy << std::endl;
                return potential_energy + kinetic_energy;
            }
        public:
            explicit GeneralizedHMCKernel(GaugeField& U_in, GaugeField& U_copy_in, GaugeField& Momentum_in, GaugeField& Momentum_copy_in, IntegratorT& Integrator_in, ActionT& Action_in, prngT& prng_in, double mixing_angle_in, double trajectory_length_in = 1.0) noexcept :
            U(U_in), U_copy(U_copy_in), Momentum(Momentum_in), Momentum_copy(Momentum_copy_in), Integrator(Integrator_in), Action(Action_in), prng(prng_in), mixing_angle(mixing_angle_in), trajectory_length(trajectory_length_in)
            {}


            bool operator()(const int n_step, const bool metropolis_step) const noexcept
            {
                // Copy old field and momentum so we can restore them in case the update gets rejected
                U_copy = U;
                Momentum_copy = Momentum;
                // Generate partially random momenta and calculate energy before time evolution
                RotateMomentum(Momentum);
                double energy_old {Hamiltonian()};
                // Perform integration with chosen integrator
                Integrator(*this, trajectory_length, n_step);
                //-----
                // Calculate energy after time evolution
                double energy_new {Hamiltonian()};
                // TODO: Probably shouldnt use a global variable for DeltaH?
                DeltaH = energy_new - energy_old;
                if (metropolis_step)
                {
                    double p {std::exp(-energy_new + energy_old)};
                    double q {prng.UniformReal()};
                    // datalog << "DeltaH: " << DeltaH << std::endl;
                    if (q <= p)
                    {
                        // TODO: Use different counter than for conventional HMC?
                        acceptance_count_hmc += 1;
                        return true;
                    }
                    else
                    {
                        U = U_copy;
                        Momentum = Momentum_copy;
                        ReverseMomenta(Momentum);
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

#endif // LETTUCE_GHMC_GAUGE_HPP
