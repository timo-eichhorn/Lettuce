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
#include <vector>
//----------------------------------------
// Standard C headers
// ...

// We can reuse the standard integrators defined in hmc_gauge.hpp, but the multiple timescale integrators below specifically require an UpdateMetadynamicsMomenta() member function

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

// std::chrono::duration<double> smearing_time;
// std::chrono::duration<double> clover_time;
// std::chrono::duration<double> deriv_time;
// std::chrono::duration<double> cderiv_time;
// std::chrono::duration<double> recursion_time;
// std::chrono::duration<double> momentum_update_time;
// std::chrono::duration<double> fields_update_time;

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
    //       Should probably create a parameter either in HMCMetaDData or the kernel itself
    template<typename IntegratorT, typename ActionT, typename prngT, typename BiasPotentialT> //requires(std::same_as<BiasPotentialT, MetaBiasPotential> or std::same_as<BiasPotentialT, VariationalBiasPotential>)
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
            BiasPotentialT& Metapotential;
            // FullTensor                                             Clover;
            // GaugeFieldSmeared                                      SmearedFields;
            // GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants> Exp_consts;
            // GaugeField                                             ForceFatLink;
            HMCMetaDData&      MetadynamicsData;
        public:
            double             trajectory_length;
            double             rho_stout_cv;
        private:
            // struct EmptyCVPathBufferT {};
            // using  CVPathBufferT = std::conditional_t<metadynamics_path_update_enabled, std::vector<double>, EmptyCVPathBufferT>;
            // [[no_unique_address]] CVPathBufferT cv_path_samples;
            using CVPathBufferT = std::vector<double>;
            CVPathBufferT cv_path_samples;
            // CVPathBufferT cv_path_actions;

            static_assert(metadynamics_path_update_enabled ? IntegratorT::path_update_compatible : true, "Path updates are only compatible with kick-drift-kick integrators");

            // The integrator needs to access the private member functions UpdateMomenta() and UpdateFields()
            friend IntegratorT;

            double MetaCharge() noexcept
            {
                MetadynamicsData.SmearedFields[0] = U;
                for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
                {
                    StoutSmearing4D(MetadynamicsData.SmearedFields[smear_count], MetadynamicsData.SmearedFields[smear_count + 1], rho_stout_cv);
                }
                return TopChargeClover(MetadynamicsData.SmearedFields[n_smear_meta]);
            }

            double MetaChargeWithConstants() noexcept
            {
                MetadynamicsData.SmearedFields[0] = U;
                for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
                {
                    StoutSmearing4DWithConstants(MetadynamicsData.SmearedFields[smear_count], MetadynamicsData.SmearedFields[smear_count + 1], MetadynamicsData.Exp_consts[smear_count], rho_stout_cv);
                }
                // Calculate clover term and topological charge (we usually need the clover term later during the update, so better this way than directly calculating the charge)
                CalculateClover<1>(MetadynamicsData.SmearedFields[n_smear_meta], MetadynamicsData.Clover);
                return TopChargeClover(MetadynamicsData.Clover);
            }

            void RandomMomentum() const noexcept
            {
                #pragma omp parallel for collapse(omp_collapse_depth)
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
            // Reverse momenta for HMC reversibility test

            void ReverseMomenta() const noexcept
            {
                #pragma omp parallel for collapse(omp_collapse_depth)
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

            void CalculateTopologicalForce(const bool perform_submeasurement = false) noexcept
            {
                // This is the Metadynamics/fat-link contribution to the momenta
                // First we need to smear the fields n_smear_meta times and store all intermediate fields
                // auto start_smearing = std::chrono::high_resolution_clock::now();
                MetadynamicsData.SmearedFields[0] = U;
                for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
                {
                    StoutSmearing4DWithConstants(MetadynamicsData.SmearedFields[smear_count], MetadynamicsData.SmearedFields[smear_count + 1], MetadynamicsData.Exp_consts[smear_count], rho_stout_cv);
                }
                // auto end_smearing = std::chrono::high_resolution_clock::now();
                // smearing_time += end_smearing - start_smearing;
                // Now we need the derivative of the metapotential and the contribution of the clover term
                // Calculate clover term on field that was smeared the most
                // auto start_clover = std::chrono::high_resolution_clock::now();
                CalculateClover<1>(MetadynamicsData.SmearedFields[n_smear_meta], MetadynamicsData.Clover);
                // auto end_clover = std::chrono::high_resolution_clock::now();
                // clover_time += end_clover - start_clover;
                // Calculate derivative of metapotential at CV_old
                // TODO: This includes the interpolation constant. Is this correct, or do we really need (V_i + V_{i + 1})/dQ (like in 1508.07270)?
                //       We could try to use a symmetric difference V(Q + 0.5 * dq) - V(Q - 0.5 * dq), but then we have to be careful with the edges...
                // auto start_deriv = std::chrono::high_resolution_clock::now();
                double CV_old {TopChargeClover(MetadynamicsData.Clover)};
                double potential_derivative {Metapotential.ReturnDerivative(CV_old)};
                if constexpr(metadynamics_path_update_enabled)
                {
                    if (perform_submeasurement)
                    {
                        cv_path_samples.push_back(CV_old);
                        // cv_path_actions.push_back(Hamiltonian() + Metapotential.ReturnPotential(CV_old));
                    }
                }
                // auto end_deriv = std::chrono::high_resolution_clock::now();
                // deriv_time += end_deriv - start_deriv;
                // Calculate clover derivative
                // auto start_cderiv = std::chrono::high_resolution_clock::now();
                #pragma omp parallel for collapse(omp_collapse_depth)
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
                // auto end_cderiv = std::chrono::high_resolution_clock::now();
                // cderiv_time += end_cderiv - start_cderiv;
                // std::cout << "Clover derivative:\n" << ForceFatLink({4,2,6,7,1}) << std::endl;
                // std::cout << "Momenta (Clover derivative) lie in algebra: " << SU3::Tests::Testsu3All(ForceFatLink, 1e-12) << std::endl;
                // Finally perform the stout force recursion
                // Exp is calculated inside the StoutForceRecrusion function, we only need to pass an array of fitting shape
                // auto start_recursion = std::chrono::high_resolution_clock::now();
                for (int smear_count = n_smear_meta; smear_count > 0; --smear_count)
                {
                    StoutForceRecursion(MetadynamicsData.SmearedFields[smear_count - 1], MetadynamicsData.SmearedFields[smear_count], MetadynamicsData.ForceFatLink, MetadynamicsData.Exp_consts[smear_count - 1], rho_stout_cv);
                    // std::cout << ForceFatLink({4,2,6,7,1}) << std::endl;
                }
                // auto end_recursion = std::chrono::high_resolution_clock::now();
                // recursion_time += end_recursion - start_recursion;
            }

            //-----
            // Update momenta for HMC

            void UpdateMomenta(const floatT epsilon, const bool perform_submeasurement = false) noexcept
            {
                CalculateTopologicalForce(perform_submeasurement);
                // auto start_momentum_update = std::chrono::high_resolution_clock::now();
                #pragma omp parallel for collapse(omp_collapse_depth)
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
                // auto end_momentum_update = std::chrono::high_resolution_clock::now();
                // momentum_update_time += end_momentum_update - start_momentum_update;
                // std::cout << "Momenta lie in algebra: " << SU3::Tests::Testsu3All(Momentum, 1e-12) << std::endl;
            }

            //-----
            // Partial momentum update functions to be used with multiple timescale integrators

            void UpdateGaugeMomenta(const floatT epsilon) noexcept
            {
                #pragma omp parallel for collapse(omp_collapse_depth)
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
                #pragma omp parallel for collapse(omp_collapse_depth)
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
                // auto start_fields_update = std::chrono::high_resolution_clock::now();
                #pragma omp parallel for collapse(omp_collapse_depth)
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
                // auto end_fields_update = std::chrono::high_resolution_clock::now();
                // fields_update_time += end_fields_update - start_fields_update;
                // std::cout << "new test: " << TestSU3All(U, 1e-8) << std::endl;
                // std::cout << "new test: " << U[1][3][4][7][2].determinant() << "\n" << U[1][3][4][7][2] * U[1][3][4][7][2].adjoint() << "\n" << std::endl;
                // std::cout << "Fields lie in group: " << SU3::TestSU3All(U, 1e-12) << std::endl;
            }

            [[nodiscard]]
            double Hamiltonian() const noexcept
            {
                double potential_energy {Action.Action(U)};
                double kinetic_energy   {0.0};
                #pragma omp parallel for collapse(omp_collapse_depth) reduction(+: kinetic_energy)
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

            void LogCVData() const
            {
                std::ofstream hmclog;
                hmclog.open(hmclogfilepath, std::fstream::out | std::fstream::app);
                std::copy(std::cbegin(cv_path_samples), std::prev(std::cend(cv_path_samples)), std::ostream_iterator<double>(hmclog, " "));
                hmclog << cv_path_samples.back() << "\n";
                // std::copy(std::cbegin(cv_path_actions), std::prev(std::cend(cv_path_actions)), std::ostream_iterator<double>(hmclog, " "));
                // hmclog << cv_path_actions.back() << "\n";
            }
        public:
            explicit HMCMetaDKernel(GaugeField& U_in, GaugeField& U_copy_in, GaugeField& Momentum_in, BiasPotentialT& Metapotential_in, HMCMetaDData& MetadynamicsData_in, IntegratorT& Integrator_in, ActionT& Action_in, prngT& prng_in, double trajectory_length_in, double rho_stout_cv_in) noexcept :
            // U(U_in), U_copy(U_copy_in), Momentum(Momentum_in), Metapotential(Metapotential_in), MetadynamicsData(MetadynamicsData_in), Integrator(Integrator_in), Action(Action_in), prng(prng_in), SmearedFields(n_smear_meta + 1), Exp_consts(n_smear_meta)
            U(U_in), U_copy(U_copy_in), Momentum(Momentum_in), Integrator(Integrator_in), Action(Action_in), prng(prng_in), Metapotential(Metapotential_in), MetadynamicsData(MetadynamicsData_in), trajectory_length(trajectory_length_in), rho_stout_cv(rho_stout_cv_in)
            {}

            bool operator()(const int n_step, const bool metropolis_step) noexcept
            {
                // Copy old field so we can restore it in case the update gets rejected
                U_copy = U;
                // Generate random momenta and calculate energy before time evolution
                RandomMomentum();
                double CV_old     {MetaCharge()};
                double energy_old {Hamiltonian() + Metapotential.ReturnPotential(CV_old)};

                // Use reserve instead of resize so we can use push_back inside the loop
                // That way there is no need to explicitly track the current step number
                if constexpr(metadynamics_path_update_enabled)
                {
                    cv_path_samples.clear();
                    cv_path_samples.reserve(n_step);
                    // cv_path_actions.clear();
                    // cv_path_actions.reserve(n_step);
                }
                else
                {
                    cv_path_samples.resize(1);
                    // Unused when path updates are disabled, no need to resize
                    // cv_path_actions.resize(1);
                }

                // Perform integration with chosen integrator
                Integrator(*this, trajectory_length, n_step);

                //-----
                // Calculate energy after time evolution
                double CV_new     {MetaCharge()};
                double energy_new {Hamiltonian() + Metapotential.ReturnPotential(CV_new)};
                // TODO: Probably shouldnt use a global variable for DeltaH?
                DeltaH = energy_new - energy_old;
                // std::cout << "Time for smearing:          " << smearing_time.count() << std::endl;
                // std::cout << "Time for clover:            " << clover_time.count() << std::endl;
                // std::cout << "Time for derivative:        " << deriv_time.count() << std::endl;
                // std::cout << "Time for clover derivative: " << cderiv_time.count() << std::endl;
                // std::cout << "Time for stout recursion:   " << recursion_time.count() << std::endl;
                // std::cout << "Time for momentum updates:  " << momentum_update_time.count() << std::endl;
                // std::cout << "Time for fields updates:    " << fields_update_time.count() << std::endl;
                // std::cout << "============================\n" << std::endl;

                // TODO: Does not yet properly work with metapotential_update_stride?
                //       Case 0: No updates at all
                //       Case 1: Regular updates
                //       Case rest: Not handled yet
                // TODO: Probably too annoying to make exact for generic integrators (merged momentum updates)...
                // if constexpr(metapotential_update_stride >= 1 and metadynamics_path_update_enabled)
                // {
                //     if (metropolis_step)
                //     {
                //         for (std::size_t i = 0; i < cv_path_samples.size(); ++i)
                //         {
                //             // Perform accept-reject for each component, if rejected replace with original CV_old
                //             bool accepted = (prng.UniformReal() <= std::exp(-cv_path_actions[i] + energy_old));
                //             cv_path_samples[i] = (accepted ? cv_path_samples[i] : CV_old);
                //         }
                //     }
                // }

                // Always write the visited CV values to a file
                LogCVData();

                if (metropolis_step)
                {
                    double p        {std::exp(-energy_new + energy_old)};
                    double q        {prng.UniformReal()};
                    bool   accepted {q <= p};
                    // datalog << "DeltaH: " << DeltaH << std::endl;
                    if (accepted)
                    {
                        Metapotential.SetCV_current(CV_new);
                        // TODO: Implement support for metapotential_update_stride > 1! Maybe best to move to bias potential class itself
                        if constexpr(metapotential_update_stride >= 1)
                        {
                            Metapotential.UpdatePotentialSymmetric(cv_path_samples);
                        }
                        acceptance_count_metadynamics_hmc += 1;
                    }
                    else
                    {
                        Metapotential.SetCV_current(CV_old);
                        std::fill(cv_path_samples.begin(), cv_path_samples.end(), CV_old);
                        if constexpr(metapotential_update_stride >= 1)
                        {
                            Metapotential.UpdatePotentialSymmetric(cv_path_samples);
                        }
                        U = U_copy;
                    }
                    // If the accept-reject step is enabled, also log the CV values that are actually used for the bias potential update
                    LogCVData();
                    return accepted;
                }
                else
                {
                    Metapotential.SetCV_current(CV_new);
                    if constexpr(metapotential_update_stride >= 1)
                    {
                        Metapotential.UpdatePotentialSymmetric(cv_path_samples);
                    }
                    datalog << "DeltaH: " << DeltaH << std::endl;
                    return true;
                }
            }

            void ReversibilityTest(const int n_step) noexcept
            {
                RandomMomentum();
                double CV_old     {MetaCharge()};
                double energy_old {Hamiltonian() + Metapotential.ReturnPotential(CV_old)};

                Integrator(*this, trajectory_length, n_step);
                ReverseMomenta();
                Integrator(*this, trajectory_length, n_step);

                double CV_new     {MetaCharge()};
                double energy_new {Hamiltonian() + Metapotential.ReturnPotential(CV_new)};

                std::cout << Lettuce::Color::BoldBlue << "Reversibility test results:\n" << Lettuce::Color::Reset
                                                      << "    H_old:   " << energy_old << "\n"
                                                      << "    H_new:   " << energy_new << "\n"
                                                      << "    ∆δH:     " << energy_new - energy_old << std::endl;
            }
    };
} // namespace GaugeUpdates


#endif // LETTUCE_HMC_METAD_HPP
