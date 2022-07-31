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

//-----
// Generate random momenta for HMC

namespace HMC_MetaD
{
    double MetaCharge(const GaugeField& Gluon, GaugeFieldSmeared& SmearedFields, const int n_smear_meta, const double smear_param)
    {
        SmearedFields[0] = Gluon;
        for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
        {
            StoutSmearing4D(SmearedFields[smear_count], SmearedFields[smear_count + 1], smear_param);
        }
        return TopChargeGluonicSymm(SmearedFields[n_smear_meta]);
    }

    double MetaChargeWithConstants(const GaugeField& Gluon, GaugeFieldSmeared& SmearedFields, FullTensor& Clover, GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, const int n_smear_meta, const double smear_param)
    {
        SmearedFields[0] = Gluon;
        for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
        {
            StoutSmearing4DWithConstants(SmearedFields[smear_count], SmearedFields[smear_count + 1], Exp_consts[smear_count], smear_param);
        }
        // Calculate clover term and topological charge (we usually need the clover term later during the update, so better this way than directly calculating the charge)
        CalculateClover(SmearedFields[n_smear_meta], Clover);
        return TopChargeGluonicSymm(Clover);
    }

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
    // Calculate topological Force/fat-link contribution from the metapotential
    void CalculateTopologicalForce(GaugeField& Gluon, GaugeField& ForceFatLink, MetaBiasPotential& Metapotential, FullTensor& Clover, GaugeFieldSmeared& SmearedFields, GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, const int n_smear_meta) noexcept
    {
        // This is the Metadynamics (fat-link) contribution to the momenta
        // First we need to smear the fields n_smear_meta times and store all intermediate fields
        SmearedFields[0] = Gluon;
        for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
        {
            // TODO: Replace global variable rho_stout with parameter?
            StoutSmearing4DWithConstants(SmearedFields[smear_count], SmearedFields[smear_count + 1], Exp_consts[smear_count], rho_stout);
        }
        // Now we need the derivative of the metapotential and the contribution of the clover term
        // Calculate clover term on field that was smeared the most
        CalculateClover(SmearedFields[n_smear_meta], Clover);
        // Calculate derivative of metapotential at CV_old
        // TODO: We already have the clover term and don't need to redo the whole computation for the topological charge!
        // TODO: This includes the interpolation constant. Is this correct, or do we really only need (V_i + V_{i + 1})/dQ (like in 1508.07270)?
        //       We could try to use a center difference V(Q + 0.5 * dq) - V(Q - 0.5 * dq), but then we have to be careful with the edges...
        double CV_old {TopChargeGluonicSymm(Clover)};
        // std::cout << "CV_old: " << CV_old << std::endl;
        double potential_derivative = Metapotential.ReturnDerivative(CV_old);
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

    void UpdateMomenta(GaugeField& Gluon, GaugeField& Momentum, MetaBiasPotential& Metapotential, FullTensor& Clover, GaugeFieldSmeared& SmearedFields, GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, const int n_smear_meta, const floatT epsilon) noexcept
    {
        // std::cout << "Start of UpdateMomenta" << std::endl;
        // We need a separate array for the fat-link contribution
        static GaugeField ForceFatLink;
        CalculateTopologicalForce(Gluon, ForceFatLink, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta);

        // Update momenta
        #pragma omp parallel for
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            link_coord current_link {t, x, y, z, mu};
            // This is the usual (thin-link) contribution to the momenta
            Matrix_3x3 st {WilsonAction::Staple(Gluon, current_link)};
            Matrix_3x3 tmp {st * Gluon(current_link).adjoint() - Gluon(current_link) * st.adjoint()};
            // Update with both the thin-link and fat-link contribution at the same time
            Momentum(current_link) -= epsilon * i<floatT> * (beta / static_cast<floatT>(12.0) * (tmp - static_cast<floatT>(1.0/3.0) * tmp.trace() * Matrix_3x3::Identity()) + ForceFatLink(current_link));
            // // Now comes the fat-link contribution
            // Momentum(current_link) -= epsilon * i<floatT> * ForceFatLink(current_link);
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
        // std::cout << "Fields lie in group: " << TestSU3All(Gluon, 1e-12) << std::endl;
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
    void Leapfrog(GaugeField& Gluon, GaugeField& Momentum, MetaBiasPotential& Metapotential, FullTensor& Clover, GaugeFieldSmeared& SmearedFields, GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, const int n_smear_meta, const int n_step) noexcept
    {
        // Calculate stepsize epsilon from n_step
        floatT epsilon {static_cast<floatT>(1.0)/n_step};
        // Perform integration
        // Momentum updates are merged in the loop
        UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, 0.5 * epsilon);
        for (int step_count = 0; step_count < n_step - 1; ++step_count)
        {
            UpdateFields(Gluon, Momentum, epsilon);
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, epsilon);
        }
        UpdateFields(Gluon, Momentum, epsilon);
        UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, 0.5 * epsilon);
    }
    //-----
    // Omelyan-Mryglod-Folk second order minimum norm integrator (improved leapfrog)
    // cf. hep-lat/0505020
    // NOTE: This version doesn't use merged momentum updates and is slightly less efficient than the one below
    void OMF_2_slow(GaugeField& Gluon, GaugeField& Momentum, MetaBiasPotential& Metapotential, FullTensor& Clover, GaugeFieldSmeared& SmearedFields, GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, const int n_smear_meta, const int n_step) noexcept
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
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, alpha);
            UpdateFields(Gluon, Momentum, beta);
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, gamma);
            UpdateFields(Gluon, Momentum, beta);
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, alpha);
        }
    }
    //-----
    // Omelyan-Mryglod-Folk second order minimum norm integrator (improved leapfrog)
    // cf. hep-lat/0505020
    void OMF_2(GaugeField& Gluon, GaugeField& Momentum, MetaBiasPotential& Metapotential, FullTensor& Clover, GaugeFieldSmeared& SmearedFields, GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, const int n_smear_meta, const int n_step) noexcept
    {
        // Calculate stepsize epsilon from n_step
        floatT epsilon {static_cast<floatT>(1.0)/n_step};
        // Integrator constants
        double alpha {0.1931833275037836 * epsilon};
        double beta  {0.5 * epsilon};
        double gamma {(1.0 - 2.0 * 0.1931833275037836) * epsilon};
        // Perform integration
        // Momentum updates are merged in the loop
        UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, alpha);
        UpdateFields(Gluon, Momentum, beta);
        UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, gamma);
        UpdateFields(Gluon, Momentum, beta);
        for (int step_count = 0; step_count < n_step - 1; ++step_count)
        {
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, 2.0 * alpha);
            UpdateFields(Gluon, Momentum, beta);
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, gamma);
            UpdateFields(Gluon, Momentum, beta);
        }
        UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, alpha);
    }
    //-----
    // Omelyan-Mryglod-Folk fourth order minimum norm integrator
    // cf. hep-lat/0505020
    // NOTE: This version doesn't use merged momentum updates and is slightly less efficient than the one below
    void OMF_4_slow(GaugeField& Gluon, GaugeField& Momentum, MetaBiasPotential& Metapotential, FullTensor& Clover, GaugeFieldSmeared& SmearedFields, GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, const int n_smear_meta, const int n_step) noexcept
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
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, alpha);
            UpdateFields(Gluon, Momentum, beta);
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, gamma);
            UpdateFields(Gluon, Momentum, delta);

            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, mu);
            UpdateFields(Gluon, Momentum, nu);
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, mu);

            UpdateFields(Gluon, Momentum, delta);
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, gamma);
            UpdateFields(Gluon, Momentum, beta);
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, alpha);
        }
    }
    //-----
    // Omelyan-Mryglod-Folk fourth order minimum norm integrator
    // cf. hep-lat/0505020
    void OMF_4(GaugeField& Gluon, GaugeField& Momentum, MetaBiasPotential& Metapotential, FullTensor& Clover, GaugeFieldSmeared& SmearedFields, GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, const int n_smear_meta, const int n_step) noexcept
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
        UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, alpha);
        UpdateFields(Gluon, Momentum, beta);
        UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, gamma);
        UpdateFields(Gluon, Momentum, delta);

        UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, mu);
        UpdateFields(Gluon, Momentum, nu);
        UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, mu);

        UpdateFields(Gluon, Momentum, delta);
        UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, gamma);
        UpdateFields(Gluon, Momentum, beta);
        for (int step_count = 0; step_count < n_step - 1; ++step_count)
        {
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, 2.0 * alpha);
            UpdateFields(Gluon, Momentum, beta);
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, gamma);
            UpdateFields(Gluon, Momentum, delta);

            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, mu);
            UpdateFields(Gluon, Momentum, nu);
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, mu);

            UpdateFields(Gluon, Momentum, delta);
            UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, gamma);
            UpdateFields(Gluon, Momentum, beta);
        }
        UpdateMomenta(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, alpha);
    }

    //-----
    // HMC for pure gauge theory
    // TODO: Constraint on FuncT?
    // TODO: Add tau parameter (trajectory time) and pass to integrator functions

    template<typename FuncT>
    // bool HMCGauge(GaugeField& Gluon, GaugeField& Gluon_copy, GaugeField& Momentum, MetaBiasPotential& Metapotential, double& CV, uint_fast64_t& acceptance_count_hmc, FuncT&& Integrator, const int n_step, bool metropolis_step, std::uniform_real_distribution<floatT>& distribution_prob) noexcept
    bool HMCGauge(GaugeField& Gluon, GaugeField& Gluon_copy, GaugeField& Momentum, MetaBiasPotential& Metapotential, uint_fast64_t& acceptance_count_hmc, FuncT&& Integrator, const int n_smear_meta, const int n_step, bool metropolis_step, std::uniform_real_distribution<floatT>& distribution_prob) noexcept
    {
        // Required arrays so we don't have to recompute everything
        // Note that we do need a separate array for Sigma/the metaforce since the stout force recursion is obviously only applied to the fat-link part
        // We thus need to keep the thin-link contribution in a separate array, calculate the fat-link contribution, and add everything together in the end
        // Required for the clover derivative
        static FullTensor Clover;
        // Required for smearing (for convenience we want size n_smear_meta + 1 to be able to hold the unsmeared field and the smeared fields)
        static GaugeFieldSmeared SmearedFields(n_smear_meta + 1);
        // Required for ExpConstants that are calculated during smearing/stout force recursion
        static GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants> Exp_consts(n_smear_meta);

        // Copy old field so we can restore it in case the update gets rejected
        Gluon_copy = Gluon;
        // Generate random momenta and calculate energy before time evolution
        RandomMomentum(Momentum);
        // Calculating the CV/smeared topological charge is not that cheap, so we can reuse it in some cases (nah actually somewhat tedious due to accept-reject)
        static double CV_old;
        // CV_old = MetaChargeWithConstants(Gluon, SmearedFields, Clover, Exp_consts, n_smear_meta, rho_stout);
        CV_old = MetaCharge(Gluon, SmearedFields, n_smear_meta, rho_stout);

        double energy_old {Hamiltonian(Gluon, Momentum) + Metapotential.ReturnPotential(CV_old)};
        // double energy_diff_nonMD {-Hamiltonian(Gluon, Momentum)};

        // Perform integration with chosen integrator
        Integrator(Gluon, Momentum, Metapotential, Clover, SmearedFields, Exp_consts, n_smear_meta, n_step);

        // Calculate energy after time evolution
        double CV_new {MetaCharge(Gluon, SmearedFields, n_smear_meta, rho_stout)};
        // double CV_new {MetaChargeWithConstants(Gluon, SmearedFields, Clover, Exp_consts, n_smear_meta, rho_stout)};
        double energy_new {Hamiltonian(Gluon, Momentum) + Metapotential.ReturnPotential(CV_new)};
        // energy_diff_nonMD += Hamiltonian(Gluon, Momentum);
        // std::cout << "Energy difference (without MetaD): " << energy_diff_nonMD << std::endl;
        // std::cout << "Energy difference (MetaD): " << energy_new - energy_old - energy_diff_nonMD << std::endl;
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
                CV_old = CV_new;
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
                // CV = CV_old;
                // TODO: Technically probably unnecessary, but better safe than sorry for now (it's cheap anyways)
                //       It's one write Michael, how much could it cost? 10 dollars?
                Metapotential.SetCV_current(CV_old);
                Gluon = Gluon_copy;
                return false;
            }
        }
        else
        {
            CV_old = CV_new;
            Metapotential.SetCV_current(CV_new);
            if constexpr(metapotential_updated)
            {
                Metapotential.UpdatePotential(CV_new);
            }
            datalog << "DeltaH: " << DeltaH << std::endl;
            return true;
        }
    }
}

#endif // LETTUCE_HMC_METAD_HPP
