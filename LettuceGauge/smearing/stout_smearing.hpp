#ifndef LETTUCE_STOUT_SMEARING_HPP
#define LETTUCE_STOUT_SMEARING_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../math/su3.hpp"
#include "../math/su3_exp.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <complex>
//----------------------------------------
// Standard C headers
#include <cmath>

// To calculate the stout force, we need the following arrays:
// 1 array holding the unsmeared gauge field
// n_smear arrays holding the smeared gauge fields (alternatively one larger array/class)
// 2 arrays holding Sigma and Sigma' (is 1 array enough if we overwrite it in place?)
// 1 array holding Lambda

// void CalculateB(const Matrix_3x3& Mat, const std::complex<floatT>& B_1, const std::complex<floatT>& B_2) noexcept
// {
//     // Matrix squared (used multiple times below, so compute once)
//     Matrix_3x3           Mat2    {Mat * Mat};
//     // Determinant and trace of a hermitian matrix are real
//     floatT               c0      {static_cast<floatT>(1.0/3.0) * std::real((Mat * Mat2).trace())};
//     floatT               c1      {static_cast<floatT>(0.5) * std::real(Mat2.trace())};
//     std::complex<floatT> c0_max  {static_cast<floatT>(2.0) * std::pow(c1 / static_cast<floatT>(3.0), static_cast<floatT>(1.5))};
//     std::complex<floatT> theta   {std::acos(c0/c0_max)};
//     std::complex<floatT> u       {std::sqrt(static_cast<floatT>(1.0/3.0) * c1) * std::cos(static_cast<floatT>(1.0/3.0) * theta)};
//     std::complex<floatT> w       {std::sqrt(c1) * std::sin(static_cast<floatT>(1.0/3.0) * theta)};
//     // Auxiliary variables depending on u that get used more than once
//     std::complex<floatT> u2      {u * u};
//     std::complex<floatT> exp_miu {std::exp(-i<floatT> * u)};
//     std::complex<floatT> exp_2iu {std::exp(static_cast<floatT>(2.0) * i<floatT> * u)};
//     // Auxiliary variables depending on w that get used more than once
//     std::complex<floatT> w2      {w * w};
//     std::complex<floatT> cosw    {std::cos(w)};
//     std::complex<floatT> xi_0    {xi_0(w)};
//     std::complex<floatT> xi_1    {xi_1(w)};
//     std::complex<floatT> i_xi0   {i<floatT> * xi_0};
//     // Denominator of f_1, f_2, f_3
//     std::complex<floatT> denom   {static_cast<floatT>(1.0) / (static_cast<floatT>(9.0) * u2 - w2)};
//     // h_0, h_1, h_2 functions to be used during calculation of f_1, f_2, f_3
//     // TODO: Numerically problematic if w -> 3u -> sqrt(3)/2 as c0 -> -c0_max?
//     // Can be circumvented by using symmetry relation of f_j, but is that really necessary here?
//     // If so, we only want to check once if c0 is negative
//     std::complex<floatT> h0      {(u2 - w2) * exp_2iu + exp_miu * (static_cast<floatT>(8.0) * u2 * cosw + static_cast<floatT>(2.0) * u * i_xi0 * (static_cast<floatT>(3.0) * u2 + w2))};
//     std::complex<floatT> h1      {static_cast<floatT>(2.0) * u * exp_2iu - exp_miu * (static_cast<floatT>(2.0) * u * cosw - (static_cast<floatT>(3.0) * u2 - w2) * i_xi0)};
//     std::complex<floatT> h2      {exp_2iu - exp_miu * (cosw + static_cast<floatT>(3.0) * u * i_xi0)};
//     // Auxiliary variables that get used more than once
//     std::complex<floatT> u2 {u * u};
//     std::complex<floatT> w2 {w * w};
//     // First calculate r^i_j
//     r1_0 = static_cast<floatT>(2.0) * (u + i<floatT> * (u2 - w2)) * exp_2iu
//          + static_cast<floatT>(2.0) * exp_miu * (static_cast<floatT>(4.0) * u * (static_cast<floatT>(2.0) - i<floatT> * u) * cosw
//          + (static_cast<floatT>(9.0) * u2 + w2 - i<floatT> * u * (static_cast<floatT>(3.0) * u2 + w2)) * i_xi0);

//     r1_1 = static_cast<floatT>(2.0) * (static_cast<floatT>(1.0) + static_cast<floatT>(2.0) * i<floatT> * u) * exp_2iu + exp_miu * (-static_cast<floatT>(2.0) * (static_cast<floatT>(1.0) - i<floatT> * u) * cosw
//          + (static_cast<floatT>(6.0) * u + i<floatT> * (w2 - static_cast<floatT>(3.0) * u2)) * i_xi0);

//     r1_2 = static_cast<floatT>(2.0) * i<floatT> * exp_2iu + i<floatT> * exp_miu * (cosw - static_cast<floatT>(3.0) * (static_cast<floatT>(1.0) - i<floatT> * u) * xi_0);

//     r2_0 = -static_cast<floatT>(2.0) * exp_2iu + static_cast<floatT>(2.0) * i<floatT> * u * exp_miu * (cosw + (static_cast<floatT>(1.0) + static_cast<floatT>(4.0) * i<floatT> * u) * xi_0 + static_cast<floatT>(3.0) * u2 * xi_1);

//     r2_1 = -i<floatT> * exp_miu * (cosw + (static_cast<floatT>(1.0) + static_cast<floatT>(2.0) * i<floatT> * u) * xi_0 - static_cast<floatT>(3.0) * u2 * xi_1);

//     r2_2 = exp_miu * (xi_0 - static_cast<floatT>(3.0) * i<floatT> * u * xi_1);
//     // Calculate b_1j and b_2j (except for missing denominator that we multiply later on)
//     std::complex<floatT> b_10 {static_cast<floatT>(2.0) * u * r1_0 + (static_cast<floatT>(3.0) * u2 - w2) * r2_0 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u2 + w2) * f0};
//     std::complex<floatT> b_11 {static_cast<floatT>(2.0) * u * r1_1 + (static_cast<floatT>(3.0) * u2 - w2) * r2_1 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u2 + w2) * f1};
//     std::complex<floatT> b_12 {static_cast<floatT>(2.0) * u * r1_2 + (static_cast<floatT>(3.0) * u2 - w2) * r2_2 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u2 + w2) * f2};
//     std::complex<floatT> b_20 {r1_0 + static_cast<floatT>(3.0) * u * r2_0 - static_cast<floatT>(24.0) * u * f0};
//     std::complex<floatT> b_21 {r1_1 + static_cast<floatT>(3.0) * u * r2_1 - static_cast<floatT>(24.0) * u * f1};
//     std::complex<floatT> b_22 {r1_2 + static_cast<floatT>(3.0) * u * r2_2 - static_cast<floatT>(24.0) * u * f2};
//     // Multiply with missing denominator
//     denom *= 0.5 * denom;
//     b_10  *= denom;
//     b_11  *= denom;
//     b_12  *= denom;
//     b_20  *= denom;
//     b_21  *= denom;
//     b_22  *= denom;
//     // Calculate B_i
//     B_1 = b_10 + b_11 * Q + b_12 * Q2;
//     B_2 = b_20 + b_21 * Q + b_22 * Q2;
// }

void CalculateExpDerivativeConstants(const GaugeField& Gluon, const GaugeField4D<Nt, Nx, Ny, Nz, ExpDerivativeConstants>& ExpConsts)
{
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        ExpConsts({t, x, y, z, mu}) = ExpDerivativeConstants(Gluon({t, x, y, z, mu}));
    }
}

// Returns a traceless hermitian (not antihermitian!) matrix
Matrix_3x3 ProjectLambda(const Matrix_3x3& mat) noexcept
{
    return static_cast<floatT>(0.5) * (mat + mat.adjoint()) - static_cast<floatT>(1.0/6.0) * (mat + mat.adjoint()).trace() * Matrix_3x3::Identity();
}

void CalculateStoutForceConstants(const GaugeField& Gluon, const GaugeField& Sigma_prev, GaugeField& Lambda, GaugeField& Exp) noexcept
{
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        SU3::ExpDerivativeConstants expd_consts(Gluon({t, x, y, z, mu}));
        // CalculateB(Gluon({t, x, y, z, mu}), expd_consts.B_1, expd_consts.B_2);
        Lambda({t, x, y, z, mu}) = ProjectionLambda((Sigma_prev({t, x, y, z, mu}) * expd_consts.B_1 * Gluon({t, x, y, z, mu})).trace() * expd_consts.Mat
                                                   + (Sigma_prev({t, x, y, z, mu}) * expd_consts.B_2 * Gluon({t, x, y, z, mu})).trace() * expd_consts.Mat2
                                                   + expd_consts.f1 * Gluon({t, x, y, z, mu}) * Sigma_prev({t, x, y, z, mu})
                                                   + expd_consts.f2 * (expd_consts.Mat * Gluon({t, x, y, z, mu}) * Sigma_prev({t, x, y, z, mu}) + Gluon({t, x, y, z, mu}) * Sigma_prev({t, x, y, z, mu}) * expd_consts.Mat));
        // Since we already have expd_consts, we can also precompute the exponentials for each site
        Exp({t, x, y, z, mu}) = SU3::exp(expd_consts);
    }
}

void StoutForceRecursion(const GaugeField& Gluon, const GaugeField& Sigma, const GaugeField& Sigma_prev, const GaugeField& Lambda, const GaugeField& Exp, const floatT smear_param) noexcept
{
    // Precompute Lambda for whole lattice, since values get used multiple times
    CalculateStoutForceConstants(Gluon, Sigma_prev, Lambda, Exp);
    // Recursively calculate Sigma (stout force at smearing level n - 1) from Sigma_prev (stout force at smearing level n)
    // Here, Gluon refers to the field at smearing level n - 1
    Matrix_3x3 st;
    Matrix_3x3 A;
    Matrix_3x3 B;
    Matrix_3x3 C;
    Matrix_3x3 force_sum;

    #pragma omp parallel for private(st, A, B, C, force_sum)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        site_coord current_site {t, x, y, z};
        for (int mu = 0; mu < 4; ++mu)
        {
            link_coord current_link {t, x, y, z, mu};
            st.noalias() = WilsonAction::Staple(Gluon, current_site, mu);
            // Omega in Peardon Morningstar paper
            A.noalias() = st * Gluon_unsmeared(current_link).adjoint();
            B.noalias() = A - A.adjoint();
            C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
            // Stout force
            for (int nu = 0; nu < 4; ++nu)
            {
                // TODO: Is there a directional symmetry we can use instead of checking this?
                // Or perhaps move the directional loops outside?
                if (mu != nu)
                {
                    site_coord site_mup     = Move< 1>(current_site, mu);
                    site_coord site_nup     = Move< 1>(current_site, nu);
                    site_coord site_nud     = Move<-1>(current_site, nu);
                    site_coord site_mup_nud = Move<-1>(site_mu     , nu);

                    force_sum = Gluon (site_mup    , nu)           * Gluon (site_nup    , mu).adjoint() * Gluon (current_site, nu).adjoint()   * Lambda(current_site, nu)
                              + Gluon (site_mup_nud, nu).adjoint() * Gluon (site_nud    , mu).adjoint() * Lambda(site_nud    , mu)             * Gluon (site_nud    , nu)
                              + Gluon (site_mup_nud, nu).adjoint() * Lambda(site_mup_nud, nu)           * Gluon (site_nud    , mu).adjoint()   * Gluon (site_nud    , nu)
                              - Gluon (site_mup_nud, nu).adjoint() * Gluon (site_nud    , mu).adjoint() * Lambda(site_nud    , nu)             * Gluon (site_nud    , nu)
                              - Lambda(site_mup    , nu)           * Gluon (site_mup    , nu)           * Gluon (site_nup    , mu).adjoint()   * Gluon (current_site, nu).adjoint()
                              + Gluon (site_mup    , nu)           * Gluon (site_nup    , mu).adjoint() * Gluon (site_nup    , mu)             * Gluon (current_site, nu).adjoint();
                }
            }
            Sigma(current_link) = Sigma_prev(current_link) * Exp(current_link) + i<floatT> * st.adjoint() * Lambda(current_link)
                                    - i<floatT> * smear_param * force_sum;
        }
    }
}

// void StoutForce()
// {
//     //...
// }

// void StoutSmearing4DWithForce(const GaugeField& Gluon_unsmeared, GaugeField& Gluon_smeared, const floatT smear_param = 0.12) noexcept
// {
//     Matrix_3x3 st;
//     Matrix_3x3 A;
//     Matrix_3x3 B;
//     Matrix_3x3 C;

//     #pragma omp parallel for private(st, A, B, C)
//     for (int t = 0; t < Nt; ++t)
//     for (int x = 0; x < Nx; ++x)
//     for (int y = 0; y < Ny; ++y)
//     for (int z = 0; z < Nz; ++z)
//     {
//         for (int mu = 0; mu < 4; ++mu)
//         {
//             // Stout smearing
//             st.noalias() = WilsonAction::Staple(Gluon_unsmeared, {t, x, y, z}, mu);
//             // Omega in Peardon Morningstar paper
//             A.noalias() = st * Gluon_unsmeared({t, x, y, z, mu}).adjoint();
//             B.noalias() = A - A.adjoint();
//             C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
//             Gluon_smeared({t, x, y, z, mu}) = SU3::exp(-i<floatT> * smear_param * C) * Gluon_unsmeared({t, x, y, z, mu});
//             SU3::Projection::GramSchmidt(Gluon_smeared({t, x, y, z, mu}));
//             // Stout force
//             for (int nu = 0; nu < 4; ++nu)
//             {
//                 // TODO: Is there a directional symmetry we can use instead of checking this?
//                 // Or perhaps move the directional loops outside?
//                 if (mu != nu)
//                 {
//                     site_coord site_mup     = Move< 1>(current_site, mu);
//                     site_coord site_nup     = Move< 1>(current_site, nu);
//                     site_coord site_nud     = Move<-1>(current_site, nu);
//                     site_coord site_mup_nud = Move<-1>(site_mu     , nu);

//                     force_sum = Gluon(site_mup    , nu)           * Gluon(site_nup    , mu).adjoint() * Gluon(current_site, nu).adjoint()   * Lambda(current_site, nu)
//                               + Gluon(site_mup_nud, nu).adjoint() * Gluon(site_nud    , mu).adjoint() * Lambda(site_nud    , mu)             * Gluon(site_nud    , nu)
//                               + Gluon(site_mup_nud, nu).adjoint() * Lambda(site_mup_nud, nu)           * Gluon(site_nud    , mu).adjoint()   * Gluon(site_nud    , nu)
//                               - Gluon(site_mup_nud, nu).adjoint() * Gluon(site_nud    , mu).adjoint() * Lambda(site_nud    , nu)             * Gluon(site_nud    , nu)
//                               - Lambda(site_mup    , nu)           * Gluon(site_mup    , nu)           * Gluon(site_nup    , mu).adjoint()   * Gluon(current_site, nu).adjoint()
//                               + Gluon(site_mup    , nu)           * Gluon(site_nup    , mu).adjoint() * Gluon(site_nup    , mu)             * Gluon(current_site, nu).adjoint();
//                 }
//             }
//             Sigma({t, x, y, z, mu}) = Sigma_prev({t, x, y, z, mu}) * SU3::exp(-i<floatT> * smear_param * C) + i<floatT> * st.adjoint() * Lambda({t, x, y, z, mu})
//                                     - i<floatT> * smear_param * force_sum;
//         }
//     }
// }

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
