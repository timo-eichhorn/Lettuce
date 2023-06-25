#ifndef LETTUCE_SMD_GAUGE_HPP
#define LETTUCE_SMD_GAUGE_HPP

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

// TODO: Implement!
// In contrast to the HMC, you need to make sure that the Momentum field is not reused/overwritten by other functions,
// since the momenta are only partially refreshed at the start of a new trajectory!
namespace GaugeUpdates
{
    template<typename IntegratorT, typename ActionT, typename prngT>
    struct SMDKernel
    {
        private:
            GaugeField&  U;
            GaugeField&  U_copy;
            GaugeField&  Momentum;
            IntegratorT& Integrator;
            ActionT&     Action;
            prngT&       prng;
        public:
            double       trajectory_length;
        private:

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
                    Momentum(current_link) = static_cast<floatT>(0.5) * A;
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
                    Matrix_3x3 tmp {Action.Staple(U, current_link) * U(current_link).adjoint()};
                    Momentum(current_link) += epsilon * beta / 6.0 * SU3::Projection::Algebra(tmp);
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
                #pragma omp parallel for reduction(+: kinetic_energy)
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
            explicit SMDKernel(GaugeField& U_in, GaugeField& U_copy_in, GaugeField& Momentum_in, IntegratorT& Integrator_in, ActionT& Action_in, prngT& prng_in, double trajectory_length_in = 1.0) noexcept :
            U(U_in), U_copy(U_copy_in), Momentum(Momentum_in), Integrator(Integrator_in), Action(Action_in), prng(prng_in), trajectory_length(trajectory_length_in)
            {}


            bool operator()(const int n_step, const bool metropolis_step) const noexcept
            {
                // Copy old field so we can restore it in case the update gets rejected
                U_copy = U;
                // Generate random momenta and calculate energy before time evolution
                RandomMomentum();
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
                        acceptance_count_smd += 1;
                        return true;
                    }
                    else
                    {
                        U = U_copy;
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
