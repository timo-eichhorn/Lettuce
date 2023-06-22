#ifndef LETTUCE_HMC_METAD_HPP
#define LETTUCE_HMC_METAD_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../coords.hpp"
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
#include <random>
//----------------------------------------
// Standard C headers
// ...

// We can reuse the integrators defined in hmc_gauge.hpp

namespace Integrators::HMC
{
    struct Leapfrog_OMF_4
    {
        template<typename HMCFunctor>
        void operator()(HMCFunctor& HMC, const double trajectory_length, const int n_step) const noexcept
        {
            // Calculate stepsize epsilon from n_step
            floatT epsilon {static_cast<floatT>(trajectory_length)/n_step};
            // Integrator constants
            const double alpha {0.08398315262876693 * epsilon};
            const double beta  {0.2539785108410595 * epsilon};
            const double gamma {0.6822365335719091 * epsilon};
            const double delta {-0.03230286765269967 * epsilon};
            const double mu    {(0.5 - 0.6822365335719091 - 0.08398315262876693) * epsilon};
            const double nu    {(1.0 - 2.0 * -0.03230286765269967 - 2.0 * 0.2539785108410595) * epsilon};
            // Perform integration
            // The expensive Metadynamics momentum updates in the outer leapfrog integrator are merged in the loop
            HMC.UpdateMetadynamicsMomenta(0.5 * epsilon);
            for (int step_count = 0; step_count < n_step - 1; ++step_count)
            {
                HMC.UpdateGaugeMomenta(alpha);
                HMC.UpdateFields(beta);
                HMC.UpdateGaugeMomenta(gamma);
                HMC.UpdateFields(delta);

                HMC.UpdateGaugeMomenta(mu);
                HMC.UpdateFields(nu);
                HMC.UpdateGaugeMomenta(mu);

                HMC.UpdateFields(delta);
                HMC.UpdateGaugeMomenta(gamma);
                HMC.UpdateFields(beta);
                HMC.UpdateGaugeMomenta(alpha);

                HMC.UpdateMetadynamicsMomenta(epsilon);
            }
            HMC.UpdateGaugeMomenta(alpha);
            HMC.UpdateFields(beta);
            HMC.UpdateGaugeMomenta(gamma);
            HMC.UpdateFields(delta);

            HMC.UpdateGaugeMomenta(mu);
            HMC.UpdateFields(nu);
            HMC.UpdateGaugeMomenta(mu);

            HMC.UpdateFields(delta);
            HMC.UpdateGaugeMomenta(gamma);
            HMC.UpdateFields(beta);
            HMC.UpdateGaugeMomenta(alpha);

            HMC.UpdateMetadynamicsMomenta(0.5 * epsilon);
        }
    };

    struct OMF_2_OMF_4_slow
    {
        template<typename HMCFunctor>
        void operator()(HMCFunctor& HMC, const double trajectory_length, const int n_step) const noexcept
        {
            // Calculate stepsize epsilon from n_step
            floatT epsilon {static_cast<floatT>(trajectory_length)/n_step};
            // OMF2 Integrator constants
            const double OMF2_alpha {0.1931833275037836 * epsilon};
            const double OMF2_beta  {(1.0 - 2.0 * 0.1931833275037836) * epsilon};
            // OMF4 Integrator constants
            const double OMF4_alpha {0.08398315262876693 * epsilon};
            const double OMF4_beta  {0.2539785108410595 * epsilon};
            const double OMF4_gamma {0.6822365335719091 * epsilon};
            const double OMF4_delta {-0.03230286765269967 * epsilon};
            const double OMF4_mu    {(0.5 - 0.6822365335719091 - 0.08398315262876693) * epsilon};
            const double OMF4_nu    {(1.0 - 2.0 * -0.03230286765269967 - 2.0 * 0.2539785108410595) * epsilon};
            // Perform integration
            for (int step_count = 0; step_count < n_step; ++step_count)
            {
                // OMF2 momentum update of Metadynamics contribution
                HMC.UpdateMetadynamicsMomenta(OMF2_alpha);
                //-----
                // OMF4 integration of gauge contribution
                HMC.UpdateGaugeMomenta(OMF4_alpha);
                HMC.UpdateFields(OMF4_beta);
                HMC.UpdateGaugeMomenta(OMF4_gamma);
                HMC.UpdateFields(OMF4_delta);

                HMC.UpdateGaugeMomenta(OMF4_mu);
                HMC.UpdateFields(OMF4_nu);
                HMC.UpdateGaugeMomenta(OMF4_mu);

                HMC.UpdateFields(OMF4_delta);
                HMC.UpdateGaugeMomenta(OMF4_gamma);
                HMC.UpdateFields(OMF4_beta);
                HMC.UpdateGaugeMomenta(OMF4_alpha);
                //-----
                // OMF2 momentum update of Metadynamics contribution
                HMC.UpdateMetadynamicsMomenta(OMF2_beta);
                //-----
                // OMF4 integration of gauge contribution
                HMC.UpdateGaugeMomenta(OMF4_alpha);
                HMC.UpdateFields(OMF4_beta);
                HMC.UpdateGaugeMomenta(OMF4_gamma);
                HMC.UpdateFields(OMF4_delta);

                HMC.UpdateGaugeMomenta(OMF4_mu);
                HMC.UpdateFields(OMF4_nu);
                HMC.UpdateGaugeMomenta(OMF4_mu);

                HMC.UpdateFields(OMF4_delta);
                HMC.UpdateGaugeMomenta(OMF4_gamma);
                HMC.UpdateFields(OMF4_beta);
                HMC.UpdateGaugeMomenta(OMF4_alpha);
                //-----
                // OMF2 momentum update of Metadynamics contribution
                HMC.UpdateMetadynamicsMomenta(OMF2_alpha);
            }
        }
    };

    struct OMF_2_OMF_4
    {
        template<typename HMCFunctor>
        void operator()(HMCFunctor& HMC, const double trajectory_length, const int n_step) const noexcept
        {
            // Calculate stepsize epsilon from n_step
            floatT epsilon {static_cast<floatT>(trajectory_length)/n_step};
            // OMF2 Integrator constants
            const double OMF2_alpha {0.1931833275037836 * epsilon};
            const double OMF2_beta  {(1.0 - 2.0 * 0.1931833275037836) * epsilon};
            // OMF4 Integrator constants (note the additional factor 0.5 since the OMF4 is embedded in the OMF2 scheme, where the fields are updated with half steps!)
            const double OMF4_alpha {0.5 * 0.08398315262876693 * epsilon};
            const double OMF4_beta  {0.5 * 0.2539785108410595 * epsilon};
            const double OMF4_gamma {0.5 * 0.6822365335719091 * epsilon};
            const double OMF4_delta {0.5 * -0.03230286765269967 * epsilon};
            const double OMF4_mu    {0.5 * (0.5 - 0.6822365335719091 - 0.08398315262876693) * epsilon};
            const double OMF4_nu    {0.5 * (1.0 - 2.0 * -0.03230286765269967 - 2.0 * 0.2539785108410595) * epsilon};
            // Perform integration
            // The expensive Metadynamics momentum updates in the outer OMF2 integrator are merged in the loop
            // OMF2 momentum update of Metadynamics contribution
            HMC.UpdateMetadynamicsMomenta(OMF2_alpha);
            for (int step_count = 0; step_count < n_step - 1; ++step_count)
            {
                // OMF4 integration of gauge contribution
                HMC.UpdateGaugeMomenta(OMF4_alpha);
                HMC.UpdateFields(OMF4_beta);
                HMC.UpdateGaugeMomenta(OMF4_gamma);
                HMC.UpdateFields(OMF4_delta);

                HMC.UpdateGaugeMomenta(OMF4_mu);
                HMC.UpdateFields(OMF4_nu);
                HMC.UpdateGaugeMomenta(OMF4_mu);

                HMC.UpdateFields(OMF4_delta);
                HMC.UpdateGaugeMomenta(OMF4_gamma);
                HMC.UpdateFields(OMF4_beta);
                HMC.UpdateGaugeMomenta(OMF4_alpha);
                //-----
                // OMF2 momentum update of Metadynamics contribution
                HMC.UpdateMetadynamicsMomenta(OMF2_beta);
                //-----
                // OMF4 integration of gauge contribution
                HMC.UpdateGaugeMomenta(OMF4_alpha);
                HMC.UpdateFields(OMF4_beta);
                HMC.UpdateGaugeMomenta(OMF4_gamma);
                HMC.UpdateFields(OMF4_delta);

                HMC.UpdateGaugeMomenta(OMF4_mu);
                HMC.UpdateFields(OMF4_nu);
                HMC.UpdateGaugeMomenta(OMF4_mu);

                HMC.UpdateFields(OMF4_delta);
                HMC.UpdateGaugeMomenta(OMF4_gamma);
                HMC.UpdateFields(OMF4_beta);
                HMC.UpdateGaugeMomenta(OMF4_alpha);
                //-----
                // OMF2 momentum update of Metadynamics contribution (factor 2 due to merged momentum updates)
                HMC.UpdateMetadynamicsMomenta(2.0 * OMF2_alpha);
            }
            // OMF4 integration of gauge contribution
            HMC.UpdateGaugeMomenta(OMF4_alpha);
            HMC.UpdateFields(OMF4_beta);
            HMC.UpdateGaugeMomenta(OMF4_gamma);
            HMC.UpdateFields(OMF4_delta);

            HMC.UpdateGaugeMomenta(OMF4_mu);
            HMC.UpdateFields(OMF4_nu);
            HMC.UpdateGaugeMomenta(OMF4_mu);

            HMC.UpdateFields(OMF4_delta);
            HMC.UpdateGaugeMomenta(OMF4_gamma);
            HMC.UpdateFields(OMF4_beta);
            HMC.UpdateGaugeMomenta(OMF4_alpha);
            //-----
            // OMF2 momentum update of Metadynamics contribution
            HMC.UpdateMetadynamicsMomenta(OMF2_beta);
            //-----
            // OMF4 integration of gauge contribution
            HMC.UpdateGaugeMomenta(OMF4_alpha);
            HMC.UpdateFields(OMF4_beta);
            HMC.UpdateGaugeMomenta(OMF4_gamma);
            HMC.UpdateFields(OMF4_delta);

            HMC.UpdateGaugeMomenta(OMF4_mu);
            HMC.UpdateFields(OMF4_nu);
            HMC.UpdateGaugeMomenta(OMF4_mu);

            HMC.UpdateFields(OMF4_delta);
            HMC.UpdateGaugeMomenta(OMF4_gamma);
            HMC.UpdateFields(OMF4_beta);
            HMC.UpdateGaugeMomenta(OMF4_alpha);
            //-----
            // OMF2 momentum update of Metadynamics contribution
            HMC.UpdateMetadynamicsMomenta(OMF2_alpha);
        }
    };
} // namespace Integrators::HMC

namespace GaugeUpdates
{
    struct HMCMetaDData
    {
        FullTensor                                             Clover;
        GaugeFieldSmeared                                      SmearedFields;
        GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants> Exp_consts;
        GaugeField                                             ForceFatLink;

        HMCMetaDData(int n_smear_meta) noexcept : Clover(), SmearedFields(n_smear_meta + 1), Exp_consts(n_smear_meta), ForceFatLink()
        {}
    };

    // TODO: Currently the kernel uses the global parameter n_smear_meta
    //       Should probably create a paramter either in HMCMetaDData or the kernel itself
    template<typename IntegratorT, typename ActionT, typename prngT>
    struct HMCMetaDKernel
    {
        private:
            GaugeField&        U;
            GaugeField&        U_copy;
            GaugeField&        Momentum;
            IntegratorT&       Integrator;
            ActionT&           Action;
            prngT&             prng;
            // Metadynamics
            // double n_smear_meta;
            // double rho_stout_meta;
            MetaBiasPotential& Metapotential;
            // FullTensor                                             Clover;
            // GaugeFieldSmeared                                      SmearedFields;
            // GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants> Exp_consts;
            // GaugeField                                             ForceFatLink;
            HMCMetaDData&      MetadynamicsData;
        public:
            double             trajectory_length;
        private:

            // The integrator needs to access the private member functions UpdateMomenta() and UpdateFields()
            friend IntegratorT;

            double MetaCharge() noexcept
            {
                MetadynamicsData.SmearedFields[0] = U;
                for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
                {
                    StoutSmearing4D(MetadynamicsData.SmearedFields[smear_count], MetadynamicsData.SmearedFields[smear_count + 1], rho_stout);
                }
                return TopChargeClover(MetadynamicsData.SmearedFields[n_smear_meta]);
            }

            double MetaChargeWithConstants() noexcept
            {
                MetadynamicsData.SmearedFields[0] = U;
                for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
                {
                    StoutSmearing4DWithConstants(MetadynamicsData.SmearedFields[smear_count], MetadynamicsData.SmearedFields[smear_count + 1], MetadynamicsData.Exp_consts[smear_count], rho_stout);
                }
                // Calculate clover term and topological charge (we usually need the clover term later during the update, so better this way than directly calculating the charge)
                CalculateClover<1>(MetadynamicsData.SmearedFields[n_smear_meta], MetadynamicsData.Clover);
                return TopChargeClover(MetadynamicsData.Clover);
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
                    link_coord current_link {t, x, y, z, mu};
                    floatT     phi1         {global_prng.Gaussian(current_link)};
                    floatT     phi2         {global_prng.Gaussian(current_link)};
                    floatT     phi3         {global_prng.Gaussian(current_link)};
                    floatT     phi4         {global_prng.Gaussian(current_link)};
                    floatT     phi5         {global_prng.Gaussian(current_link)};
                    floatT     phi6         {global_prng.Gaussian(current_link)};
                    floatT     phi7         {global_prng.Gaussian(current_link)};
                    floatT     phi8         {global_prng.Gaussian(current_link)};

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
                MetadynamicsData.SmearedFields[0] = U;
                for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
                {
                    StoutSmearing4DWithConstants(MetadynamicsData.SmearedFields[smear_count], MetadynamicsData.SmearedFields[smear_count + 1], MetadynamicsData.Exp_consts[smear_count], rho_stout);
                }
                // Now we need the derivative of the metapotential and the contribution of the clover term
                // Calculate clover term on field that was smeared the most
                CalculateClover<1>(MetadynamicsData.SmearedFields[n_smear_meta], MetadynamicsData.Clover);
                // Calculate derivative of metapotential at CV_old
                // TODO: This includes the interpolation constant. Is this correct, or do we really need (V_i + V_{i + 1})/dQ (like in 1508.07270)?
                //       We could try to use a symmetric difference V(Q + 0.5 * dq) - V(Q - 0.5 * dq), but then we have to be careful with the edges...
                double CV_old {TopChargeClover(MetadynamicsData.Clover)};
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
                    MetadynamicsData.ForceFatLink(current_site, mu) = potential_derivative * CloverDerivative(MetadynamicsData.SmearedFields[n_smear_meta], MetadynamicsData.Clover, current_site, mu);
                }
                // std::cout << "Clover derivative:\n" << ForceFatLink({4,2,6,7,1}) << std::endl;
                // std::cout << "Momenta (Clover derivative) lie in algebra: " << SU3::Tests::Testsu3All(ForceFatLink, 1e-12) << std::endl;
                // Finally perform the stout force recursion
                // Exp is calculated inside the StoutForceRecrusion function, we only need to pass an array of fitting shape
                for (int smear_count = n_smear_meta; smear_count > 0; --smear_count)
                {
                    // TODO: Replace global variable rho_stout with parameter?
                    StoutForceRecursion(MetadynamicsData.SmearedFields[smear_count - 1], MetadynamicsData.SmearedFields[smear_count], MetadynamicsData.ForceFatLink, MetadynamicsData.Exp_consts[smear_count - 1], rho_stout);
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
                    Momentum(current_link) += epsilon * (beta / 6.0 * SU3::Projection::Algebra(tmp) + MetadynamicsData.ForceFatLink(current_link));
                }
                // std::cout << "Momenta lie in algebra: " << SU3::Tests::Testsu3All(Momentum, 1e-12) << std::endl;
            }

            //-----
            // Partial momentum update functions to be used with multiple timescale integrators

            void UpdateGaugeMomenta(const floatT epsilon) noexcept
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
            }

            void UpdateMetadynamicsMomenta(const floatT epsilon) noexcept
            {
                #pragma omp parallel for
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int mu = 0; mu < 4; ++mu)
                {
                    link_coord current_link {t, x, y, z, mu};
                    Momentum(current_link) += epsilon * MetadynamicsData.ForceFatLink(current_link);
                }
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
            explicit HMCMetaDKernel(GaugeField& U_in, GaugeField& U_copy_in, GaugeField& Momentum_in, MetaBiasPotential& Metapotential_in, HMCMetaDData& MetadynamicsData_in, IntegratorT& Integrator_in, ActionT& Action_in, prngT& prng_in, double trajectory_length_in = 1.0) noexcept :
            // U(U_in), U_copy(U_copy_in), Momentum(Momentum_in), Metapotential(Metapotential_in), MetadynamicsData(MetadynamicsData_in), Integrator(Integrator_in), Action(Action_in), prng(prng_in), SmearedFields(n_smear_meta + 1), Exp_consts(n_smear_meta)
            U(U_in), U_copy(U_copy_in), Momentum(Momentum_in), Metapotential(Metapotential_in), MetadynamicsData(MetadynamicsData_in), Integrator(Integrator_in), Action(Action_in), prng(prng_in), trajectory_length(trajectory_length_in)
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
                Integrator(*this, trajectory_length, n_step);
                //-----
                // Calculate energy after time evolution
                double CV_new     {MetaCharge()};
                double energy_new {Hamiltonian() + Metapotential.ReturnPotential(CV_new)};
                // TODO: Probably shouldnt use a global variable for DeltaH?
                DeltaH = energy_new - energy_old;
                if (metropolis_step)
                {
                    double p {std::exp(-energy_new + energy_old)};
                    double q {prng.UniformReal()};
                    // datalog << "DeltaH: " << DeltaH << std::endl;
                    if (q <= p)
                    {
                        Metapotential.SetCV_current(CV_new);
                        if constexpr(metapotential_updated)
                        {
                            Metapotential.UpdatePotential(CV_new);
                            Metapotential.UpdatePotential(-CV_new);
                        }
                        acceptance_count_metadynamics_hmc += 1;
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
                        Metapotential.UpdatePotential(-CV_new);
                    }
                    datalog << "DeltaH: " << DeltaH << std::endl;
                    return true;
                }
            }
    };
} // namespace GaugeUpdates


#endif // LETTUCE_HMC_METAD_HPP
