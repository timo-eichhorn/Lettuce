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
// Stout smearing of gluon fields in all 4 directions

void StoutSmearing4D(const GaugeField& Gluon_unsmeared, GaugeField& Gluon_smeared, const floatT smear_param = 0.12)
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
            Sigma.noalias() = WilsonAction::Staple(Gluon_unsmeared, current_link);
            A.noalias() = Sigma * Gluon_unsmeared(current_link).adjoint();
            // TODO: Replace with projector function?
            B.noalias() = A - A.adjoint();
            C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
            // Cayley-Hamilton exponential
            // Gluon_smeared(current_link) = SU3::exp(-i<floatT> * smear_param * C) * Gluon_unsmeared(current_link);
            // Eigen exponential (Scaling and squaring)
            Gluon_smeared(current_link) = (smear_param * C).exp() * Gluon_unsmeared(current_link);
            // Gluon_smeared[t][x][y][z][mu] = CayleyMap(i<floatT> * smear_param * C) * Gluon_unsmeared[t][x][y][z][mu];
            SU3::Projection::GramSchmidt(Gluon_smeared(current_link));
        }
    }
}

void StoutSmearing4DStable(const GaugeField& Gluon_unsmeared, GaugeField& Gluon_smeared, const floatT smear_param = 0.12)
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
            Sigma.noalias() = WilsonAction::Staple(Gluon_unsmeared, current_link);
            A.noalias() = Sigma * Gluon_unsmeared(current_link).adjoint();
            // TODO: Replace with projector function?
            B.noalias() = A - A.adjoint();
            C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
            // Cayley-Hamilton exponential
            // Gluon_smeared(current_link) = SU3::exp(-i<floatT> * smear_param * C) * Gluon_unsmeared(current_link);
            // Eigen exponential (Scaling and squaring)
            Gluon_smeared(current_link) = (smear_param * C).exp() * Gluon_unsmeared(current_link);
            SU3::Projection::GramSchmidt(Gluon_smeared(current_link));
        }
    }
}

// [[nodiscard]]
// SmearedFieldTuple StoutSmearingN(GaugeField& Gluon1, GaugeField& Gluon2, const int n_smear, const floatT smear_param = 0.12)
// {
//     for (int smear_count = 0; smear_count < n_smear; ++smear_count)
//     {
//         if (smear_count % 2 == 0)
//         {
//             StoutSmearing4D(Gluon1, Gluon2, smear_param);
//         }
//         else
//         {
//             StoutSmearing4D(Gluon2, Gluon1, smear_param);
//         }
//     }
//     if (n_smear % 2 == 0)
//     {
//         return {Gluon1, Gluon2};
//     }
//     else
//     {
//         return {Gluon2, Gluon1};
//     }
// }

// TODO: This is potentially dangerous, since we need to make sure we use the correct Gluon array afterwards,
//       which depends on n_smear. For even n_smear, we need to use Gluon1, for odd n_smear we need to use Gluon2!

void StoutSmearingN(GaugeField& Gluon1, GaugeField& Gluon2, const int N, const floatT smear_param = 0.12)
{
    for (int smear_count = 0; smear_count < N; ++smear_count)
    {
        if (smear_count % 2 == 0)
        {
            StoutSmearing4D(Gluon1, Gluon2, smear_param);
        }
        else
        {
            StoutSmearing4D(Gluon2, Gluon1, smear_param);
        }
    }
}

// void StoutSmearingN(GaugeFieldSmeared& SmearedFields, const int offset, const int n_smear, const floatT smear_param = 0.12)
// {
//     for (int smear_count = 0; smear_count < n_smear; ++smear_count)
//     {
//         StoutSmearing4D(SmearedFields[(offset + smear_count) % 2], SmearedFields[(offset + smear_count + 1) % 2], smear_param);
//     }
// }

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

void StoutSmearing4DWithConstants(const GaugeField& Gluon_unsmeared, GaugeField& Gluon_smeared, GaugeField4D<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, const floatT smear_param = 0.12) noexcept
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
            st.noalias() = WilsonAction::Staple(Gluon_unsmeared, current_link);
            A.noalias() = st * Gluon_unsmeared(current_link).adjoint();
            // TODO: Replace with projector function?
            B.noalias() = A - A.adjoint();
            C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
            // We want the ExpConstants so we can later reuse them during the calculation of the stout force recursion
            Exp_consts(current_link) = SU3::ExpConstants(-i<floatT> * smear_param * C);
            Gluon_smeared(current_link) = SU3::exp(Exp_consts(current_link)) * Gluon_unsmeared(current_link);
            SU3::Projection::GramSchmidt(Gluon_smeared(current_link));
        }
    }
}

// Calculate Lambda
void CalculateLambda(const GaugeField& Gluon, const GaugeField& Sigma, const GaugeField4D<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, GaugeField& Lambda, const floatT smear_param) noexcept
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
        // // Calculate argument of exponential function
        // st.noalias() = WilsonAction::Staple(Gluon, current_link);
        // // Omega in Peardon Morningstar paper
        // A.noalias() = st * Gluon(current_link).adjoint();
        // B.noalias() = A - A.adjoint();
        // C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
        // SU3::ExpDerivativeConstants expd_consts(-i<floatT> * smear_param * C);
        
        // Since we have already calculated Exp_consts, we can reuse quite some stuff
        SU3::ExpDerivativeConstants expd_consts(Exp_consts(current_link));
        // tmp is reused multiple times below, so precompute (compared to the Peardon Morningstar paper, we shuffled the expressions inside the first two traces around using the cyclicity)
        // tmp = U_{mu}(n) * Sigma'_{mu}(n)
        Matrix_3x3 tmp {Gluon(current_link) * Sigma(current_link)};
        // // Calculate Lambda (used during stout force recursion)
        Lambda(current_link) = SU3::Projection::TracelessHermitian((expd_consts.B_1 * tmp).trace() * expd_consts.Mat
                                                                 + (expd_consts.B_2 * tmp).trace() * expd_consts.Mat2
                                                                 +  expd_consts.f1  * tmp
                                                                 +  expd_consts.f2  * (expd_consts.Mat * tmp + tmp * expd_consts.Mat));
        // Since we already have expd_consts, we can also precompute the exponentials for each site (also used during stout force recursion)
        // Exp(current_link) = SU3::exp(expd_consts);
    }
}

void StoutForceRecursion(const GaugeField& Gluon, const GaugeField& Gluon_prev, GaugeField& Sigma, const GaugeField4D<Nt, Nx, Ny, Nz, SU3::ExpConstants>& Exp_consts, const floatT smear_param) noexcept
{
    // First multiply the incoming Sigma with V^{\dagger}
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int mu = 0; mu < 4; ++mu)
    {
        link_coord current_link {t, x, y, z, mu};
        Sigma(current_link) = Gluon_prev(current_link).adjoint() * Sigma(current_link);
    }
    // Precompute Lambda for whole lattice, since values get used multiple times
    // Exp_consts should already be known from stout smearing (if StoutSmearing4DWithConstants was used instead of the normal stout smearing function)
    static GaugeField Lambda;
    CalculateLambda(Gluon, Sigma, Exp_consts, Lambda, smear_param);
    // Recursively calculate Sigma (stout force at smearing level n - 1) from Sigma_prev (stout force at smearing level n)
    // Since we only need the local contribution from Sigma, we do not need two array for Sigma and Sigma_prev and can simply update Sigma in place
    // Lambda contains contributions from Sigma_prev, but as long as we call CalculateLambda() first everything is correct
    // Here, Gluon refers to the field at smearing level n - 1

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
                if (mu != nu)
                {
                    site_coord site_mup     = Move< 1>(current_site, mu);
                    site_coord site_nup     = Move< 1>(current_site, nu);
                    site_coord site_nud     = Move<-1>(current_site, nu);
                    site_coord site_mup_nud = Move<-1>(site_mup    , nu);

                    force_sum += Gluon (site_mup    , nu)           * Gluon (site_nup    , mu).adjoint() * Gluon (current_site, nu).adjoint()   * Lambda(current_site, nu)
                               + Gluon (site_mup_nud, nu).adjoint() * Gluon (site_nud    , mu).adjoint() * Lambda(site_nud    , mu)             * Gluon (site_nud    , nu)
                               + Gluon (site_mup_nud, nu).adjoint() * Lambda(site_mup_nud, nu)           * Gluon (site_nud    , mu).adjoint()   * Gluon (site_nud    , nu)
                               - Gluon (site_mup_nud, nu).adjoint() * Gluon (site_nud    , mu).adjoint() * Lambda(site_nud    , nu)             * Gluon (site_nud    , nu)
                               - Lambda(site_mup    , nu)           * Gluon (site_mup    , nu)           * Gluon (site_nup    , mu).adjoint()   * Gluon (current_site, nu).adjoint()
                               + Gluon (site_mup    , nu)           * Gluon (site_nup    , mu).adjoint() * Lambda(site_nup    , mu)             * Gluon (current_site, nu).adjoint();
                }
            }
            // Staple is used both during the calculation of stout force constants and below during the actual recursion, also precompute whole array?
            // st_adj.noalias() = WilsonAction::Staple(Gluon, current_link).adjoint();
            Matrix_3x3 C_adj {smear_param * WilsonAction::Staple(Gluon, current_link).adjoint()};
            // Complete recursion relation
            // Sigma(current_link) = Sigma(current_link) * SU3::exp(Exp_consts(current_link)) + i<floatT> * C_adj * Lambda(current_link)
            //                     - i<floatT> * smear_param * force_sum;
            // With projector
            // Sigma(current_link) = SU3::Projection::Algebra(Sigma(current_link) * SU3::exp(Exp_consts(current_link)) + i<floatT> * C_adj * Lambda(current_link)
            //                                              - i<floatT> * smear_param * force_sum);
            // Multiply with U and the apply traceless hermitian projector
            Sigma(current_link) = SU3::Projection::Algebra(Gluon(current_link) * Sigma(current_link) * SU3::exp(Exp_consts(current_link)) + i<floatT> * Gluon(current_link) * C_adj * Lambda(current_link)
                                                         - i<floatT> * smear_param * Gluon(current_link) * force_sum);
            // Sigma(current_link) = C_adj * Lambda(current_link);
            // Sigma(current_link) = i<floatT> * C_adj * Lambda(current_link);
        }
    }
    // std::cout << Sigma(1, 2, 0, 1, 3) << std::endl;
    // std::cout << "Sigma lies in algebra: " << SU3::Tests::Testsu3All(Sigma) << std::endl;
    // std::cout << "Sigma lies in group:   " << SU3::Tests::TestSU3All(Sigma) << std::endl;
}

// TODO: Implement smearing as functor
// template<typename GaugeActionT>
// struct StoutSmearingKernel
// {
//     private:
//         floatT smear_param;
//     public:
//         explicit StoutSmearingKernel(const floatT smear_param_in) noexcept :
//         smear_param(smear_param_in)
//         {}

//         void operator()(const )
//         {
//             Matrix_3x3 st {WilsonAction::Staple()}
//         }
// };

#endif // LETTUCE_STOUT_SMEARING_HPP
