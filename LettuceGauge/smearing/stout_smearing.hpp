#ifndef LETTUCE_STOUT_SMEARING_HPP
#define LETTUCE_STOUT_SMEARING_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../actions/gauge/wilson_action.hpp"
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
#include <cmath>

// To calculate the stout force, we need the following arrays:
// 1 array holding the unsmeared gauge field
// n_smear arrays holding the smeared gauge fields (alternatively one larger array/class)
// 1 array holding Sigma and Sigma' (1 array is enough if we overwrite it in place)
// 1 array holding Lambda
// 1 array holding Exp

//-----
// Stout smearing of gauge fields in all 4 directions

void StoutSmearing4D(const GaugeField& U_unsmeared, GaugeField& U_smeared, const floatT smear_param = 0.12)
{
    Matrix_3x3 Sigma;
    Matrix_3x3 A;
    Matrix_3x3 B;
    Matrix_3x3 C;

    #pragma omp parallel for private(Sigma, A, B, C)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        for (int mu = 0; mu < 4; ++mu)
        {
            link_coord current_link {t, x, y, z, mu};
            Sigma.noalias() = WilsonAction::Staple(U_unsmeared, current_link);
            A.noalias() = Sigma * U_unsmeared(current_link).adjoint();
            // TODO: Replace with projector function SU3::Projection::Algebra()?
            B.noalias() = A - A.adjoint();
            C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
            // Cayley-Hamilton exponential
            U_smeared(current_link) = SU3::exp(-i<floatT> * smear_param * C) * U_unsmeared(current_link);
            // Eigen exponential (Scaling and squaring)
            // U_smeared(current_link) = (smear_param * C).exp() * U_unsmeared(current_link);
            SU3::Projection::GramSchmidt(U_smeared(current_link));
        }
    }
}

// TODO: This is potentially dangerous, since we need to make sure we use the correct U array afterwards,
//       which depends on n_smear. For even n_smear, we need to use U1, for odd n_smear we need to use U2!

void StoutSmearingN(GaugeField& U1, GaugeField& U2, const int N, const floatT smear_param = 0.12)
{
    for (int smear_count = 0; smear_count < N; ++smear_count)
    {
        if (smear_count % 2 == 0)
        {
            StoutSmearing4D(U1, U2, smear_param);
        }
        else
        {
            StoutSmearing4D(U2, U1, smear_param);
        }
    }
}

void StoutSmearingAll(GaugeFieldSmeared& SmearedFields, const floatT smear_param = 0.12) noexcept
{
    if (SmearedFields.ReturnNsmear() > 1)
    {
        for (int smear_count = 0; smear_count < SmearedFields.ReturnNsmear() - 1; ++smear_count)
        {
            StoutSmearing4D(SmearedFields[smear_count], SmearedFields[smear_count + 1], smear_param);
        }
    }
    else
    {
        std::cerr << "Can't smear field with NSmear <= 1!" << std::endl;
    }
}

void StoutSmearing4DWithConstants(const GaugeField& U_unsmeared, GaugeField& U_smeared, GaugeField4D<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, const floatT smear_param = 0.12) noexcept
{
    Matrix_3x3 st;
    Matrix_3x3 A;
    Matrix_3x3 B;
    Matrix_3x3 C;

    #pragma omp parallel for private(st, A, B, C)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        for (int mu = 0; mu < 4; ++mu)
        {
            link_coord current_link {t, x, y, z, mu};
            st.noalias() = WilsonAction::Staple(U_unsmeared, current_link);
            A.noalias() = st * U_unsmeared(current_link).adjoint();
            // TODO: Replace with projector function SU3::Projection::Algebra()?
            B.noalias() = A - A.adjoint();
            C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
            // We want the ExpConstants so we can later reuse them during the calculation of the stout force recursion
            Exp_consts(current_link) = SU3::ExpConstants(-i<floatT> * smear_param * C);
            U_smeared(current_link) = SU3::exp(Exp_consts(current_link)) * U_unsmeared(current_link);
            SU3::Projection::GramSchmidt(U_smeared(current_link));
        }
    }
}

// Calculate Lambda
void CalculateLambda(const GaugeField& U, const GaugeField& Sigma, const GaugeField4D<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, GaugeField& Lambda, const floatT smear_param) noexcept
{
    Matrix_3x3 st;
    Matrix_3x3 A;
    Matrix_3x3 B;
    Matrix_3x3 C;

    #pragma omp parallel for private(st, A, B, C)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int mu = 0; mu < 4; ++mu)
    {
        link_coord current_link {t, x, y, z, mu};
        // Since we have already calculated Exp_consts, we can reuse quite some stuff
        SU3::ExpDerivativeConstants expd_consts(Exp_consts(current_link));
        // tmp is reused multiple times below, so precompute (compared to the Peardon Morningstar paper, we shuffled the expressions inside the first two traces around using the cyclicity)
        // tmp = U_{mu}(n) * Sigma'_{mu}(n)
        Matrix_3x3 tmp {U(current_link) * Sigma(current_link)};
        // Calculate Lambda (used during stout force recursion)
        Lambda(current_link) = SU3::Projection::TracelessHermitian((expd_consts.B_1 * tmp).trace() * expd_consts.Mat
                                                                 + (expd_consts.B_2 * tmp).trace() * expd_consts.Mat2
                                                                 +  expd_consts.f1  * tmp
                                                                 +  expd_consts.f2  * (expd_consts.Mat * tmp + tmp * expd_consts.Mat));
        // Since we already have expd_consts, we can also precompute the exponentials for each site (also used during stout force recursion)
        // Exp(current_link) = SU3::exp(expd_consts);
    }
}

void StoutForceRecursion(const GaugeField& U, const GaugeField& U_prev, GaugeField& Sigma, const GaugeField4D<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, const floatT smear_param) noexcept
{
    // In the original Peardon & Morningstar paper, the Sigma they work with is not the algebra-valued force F
    // Instead, the relation between Sigma and F is given by:
    // Sigma' = U'^{\dagger} F (all at the same smearing level)
    // Since we usually work with F, we first need to multiply the incoming 'Sigma' with U'^{\dagger} to actually obtain the Sigma from the Peardon & Morningstar paper
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int mu = 0; mu < 4; ++mu)
    {
        link_coord current_link {t, x, y, z, mu};
        Sigma(current_link) = U_prev(current_link).adjoint() * Sigma(current_link);
    }
    // Precompute Lambda for whole lattice, since values get used multiple times
    // Exp_consts should already be known from stout smearing (if StoutSmearing4DWithConstants was used instead of the normal stout smearing function)
    static GaugeField Lambda;
    CalculateLambda(U, Sigma, Exp_consts, Lambda, smear_param);
    // Recursively calculate Sigma (stout force at smearing level n - 1) from Sigma_prev (stout force at smearing level n)
    // Since we only need the local contribution from Sigma, we do not need two arrays for Sigma and Sigma_prev and can simply update Sigma in place
    // Lambda contains contributions from Sigma_prev, but as long as we call CalculateLambda() first everything is correct
    // Here, U refers to the field at smearing level n - 1, and U_prev to the field at smearing level n

    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        site_coord current_site {t, x, y, z};
        for (int mu = 0; mu < 4; ++mu)
        {
            link_coord current_link {t, x, y, z, mu};
            Matrix_3x3 force_sum {Matrix_3x3::Zero()};
            for (int nu = 0; nu < 4; ++nu)
            {
                // TODO: Perhaps move the directional loops outside?
                // TODO: Include generic rho_{mu, nu} for anisotropic smearing?
                if (mu != nu)
                {
                    site_coord site_mup     = U.Move< 1>(current_site, mu);
                    site_coord site_nup     = U.Move< 1>(current_site, nu);
                    site_coord site_nud     = U.Move<-1>(current_site, nu);
                    site_coord site_mup_nud = U.Move<-1>(site_mup    , nu);

                    // Original version without reusing common factors (left here due to better readability)
                    // force_sum += U(     site_mup, nu)           * U(         site_nup, mu).adjoint() * U( current_site, nu).adjoint() * Lambda(current_site, nu)
                    //            + U( site_mup_nud, nu).adjoint() * U(         site_nud, mu).adjoint() * Lambda(site_nud, mu)           * U(         site_nud, nu)
                    //            + U( site_mup_nud, nu).adjoint() * Lambda(site_mup_nud, nu)           * U(     site_nud, mu).adjoint() * U(         site_nud, nu)
                    //            - U( site_mup_nud, nu).adjoint() * U(         site_nud, mu).adjoint() * Lambda(site_nud, nu)           * U(         site_nud, nu)
                    //            - Lambda(site_mup, nu)           * U(         site_mup, nu)           * U(     site_nup, mu).adjoint() * U(     current_site, nu).adjoint()
                    //            + U(     site_mup, nu)           * U(         site_nup, mu).adjoint() * Lambda(site_nup, mu)           * U(     current_site, nu).adjoint();

                    // Calculate terms occuring more than once
                    Matrix_3x3 tmp1 {U(    site_mup, nu)       * U(site_nup, mu).adjoint()                      };                                                                                          // Appears in rows 1, 5, and 6 (TODO?: Technically Matrix_SU3)
                    Matrix_3x3 tmp2 {U(site_nud, mu).adjoint() * ( Lambda(site_nud, mu) - Lambda(site_nud, nu) )};                                                                                          // Appears in rows 2 and 4
                    // Less readable, but more performant version
                    force_sum += (tmp1                          * ( U(current_site , nu).adjoint()  * Lambda(current_site, nu) + Lambda(site_nup, mu) * U(current_site, nu).adjoint() )                     // Rows 1 and 6 in equation C6 in [2307.04742]
                                + U(site_mup_nud, nu).adjoint() * ( tmp2 + Lambda(site_mup_nud, nu) * U(site_nud, mu).adjoint()                                                       ) * U(site_nud, nu)   // Rows 2, 4, and 3 in equation C6 in [2307.04742]
                                - Lambda(site_mup, nu)          *   tmp1                            * U(current_site, nu).adjoint()                                                                      ); // Row 5 in equation C6 in [2307.04742]

                    // TODO: In the future if we want to support anisotropic smearing we need to replace the force sum above with the following expression
                    // Matrix_3x3 tmp1 {U(    site_mup, nu)       * U(site_nup, mu).adjoint()                                              };                                                                                           // Appears in rows 1, 5, and 6
                    // Matrix_3x3 tmp2 {U(site_nud, mu).adjoint() * ( rho_mu_nu * Lambda(site_nud, mu) - rho_nu_mu * Lambda(site_nud, nu) )};                                                                                           // Appears in rows 2 and 4
                    // force_sum += (tmp1                          * ( rho_nu_mu * U(current_site , nu).adjoint()  * Lambda(current_site, nu) + rho_mu_nu *  Lambda(site_nup, mu) * U(current_site, nu).adjoint() )                     // Rows 1 and 6 in equation C6 in [2307.04742]
                    //             + U(site_mup_nud, nu).adjoint() * ( tmp2 + rho_nu_mu * Lambda(site_mup_nud, nu) * U(site_nud, mu).adjoint()                                                                    ) * U(site_nud, nu)   // Rows 2, 4, and 3 in equation C6 in [2307.04742]
                    //             - rho_nu_mu                     * Lambda(site_mup, nu) * tmp1                   * U(current_site, nu).adjoint()                                                                                   ); // Row 5 in equation C6 in [2307.04742]
                }
            }
            // TODO: In the future if we want to support anisotropic smearing we need to replace the smear_param below (in both the computation of C_adj and Sigma) with the appropriate different prefactors
            // Staple is used both during the calculation of stout force constants and below during the actual recursion, also precompute whole array?
            // st_adj.noalias() = WilsonAction::Staple(U, current_link).adjoint();
            Matrix_3x3 C_adj {smear_param * WilsonAction::Staple(U, current_link).adjoint()};
            // Multiply with U and then apply traceless hermitian projector
            Sigma(current_link) = SU3::Projection::Algebra(U(current_link) * ( Sigma(current_link) * SU3::exp(Exp_consts(current_link)) + i<floatT> * C_adj * Lambda(current_link) - i<floatT> * smear_param * force_sum ));
        }
    }
    // std::cout << Sigma(1, 2, 0, 1, 3) << std::endl;
    // std::cout << "Sigma lies in algebra: " << SU3::Tests::Testsu3All(Sigma) << std::endl;
    // std::cout << "Sigma lies in group:   " << SU3::Tests::TestSU3All(Sigma) << std::endl;
}

// template<typename ActionT>
struct StoutSmearingKernel
{
    private:
        // TODO: In contrast to the gradient flow, do we want to support different integrators and actions here?
        //       If yes, correctly implementing the smearing force recursion (without some kind of automatic differentiation) will most likely be very painful
        const GaugeField&  U_unsmeared;
              GaugeField&  U_smeared;
              GaugeField&  Force;
              // IntegratorT& Integrator;
              // ActionT&     Action;
              floatT       epsilon;

        // The epsilon is only included here, not in CalculateForce, in contrast to the GradientFlowKernel
        // where we need parameters in both functions for the higher order integrators
        void UpdateFields(GaugeField& U, const GaugeField& Z) const noexcept
        {
            #pragma omp parallel for
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
            for (int mu = 0; mu < 4; ++mu)
            {
                link_coord current_link {t, x, y, z, mu};
                U(current_link) = SU3::exp(-i<floatT> * epsilon * Z(current_link)) * U(current_link);
                // Eigen exponential (Scaling and squaring)
                // U(current_link) = (epsilon * Z(current_link)).exp() * U(current_link);
                // Projection to SU(3) (necessary?)
                SU3::Projection::GramSchmidt(U(current_link));
            }
        }

        void CalculateForce(const GaugeField& U, GaugeField& Z) const noexcept
        {
            #pragma omp parallel for
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
            {
                for (int mu = 0; mu < 4; ++mu)
                {
                    link_coord current_link {t, x, y, z, mu};
                    // Matrix_3x3 st {Action.Staple(U, current_link)};
                    Matrix_3x3 st {WilsonAction::Staple(U, current_link)};
                    Matrix_3x3 A  {st * U(current_link).adjoint()};
                    Z(current_link) = SU3::Projection::Algebra(A);
                }
            }
        }
    public:
        // explicit StoutSmearingKernel(const GaugeField& U_unsmeared_in, GaugeField& U_smeared_in, ActionT& Action_in, const floatT epsilon_in) noexcept :
        // U_unsmeared(U_unsmeared_in), U_smeared(U_smeared_in), Action(Action_in), epsilon(epsilon_in)
        // {}
        explicit StoutSmearingKernel(const GaugeField& U_unsmeared_in, GaugeField& U_smeared_in, GaugeField& Force_in, const floatT epsilon_in) noexcept :
        U_unsmeared(U_unsmeared_in), U_smeared(U_smeared_in), Force(Force_in), epsilon(epsilon_in)
        {}

        void operator()(const int n_step) const noexcept
        {
            U_smeared = U_unsmeared;
            // From now on only work with U_smeared and Force
            for (int step_count = 0; step_count < n_step; ++step_count)
            {
                CalculateForce(U_smeared, Force);
                UpdateFields(U_smeared, Force);
            }
        }

        void Resume(const int n_step) const noexcept
        {
            for (int step_count = 0; step_count < n_step; ++step_count)
            {
                CalculateForce(U_smeared, Force);
                UpdateFields(U_smeared, Force);
            }
        }

        void SetEpsilon(const floatT epsilon_in) noexcept
        {
            epsilon = epsilon_in;
        }

        floatT GetEpsilon() const noexcept
        {
            return epsilon;
        }

        static std::string ReturnIntegratorName()
        {
            return "StoutSmearing";
        }
};

#endif // LETTUCE_STOUT_SMEARING_HPP
