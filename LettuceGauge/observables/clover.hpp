#ifndef LETTUCE_CLOVER_HPP
#define LETTUCE_CLOVER_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "plaquette.hpp"
#include "wilson_loop.hpp"
#include <Eigen/Dense>
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
// ...
//----------------------------------------
// Standard C headers
// ...

//+---------------------------------------------------------------------------------+
//| This file provides a function template that allows the calculation of generic   |
//| N_mu x N_nu clover term components (where component refers to the clover term   |
//| for a fixed set of Lorentz indices).                                            |
//| In addition, a function to calculate the plaquette-based clover term for all    |
//| lattice sites, as well as a function to calculate the derivative of the         |
//| plaquette-based clover term with respect to a single link are provided.         |
//+---------------------------------------------------------------------------------+

template<int N_mu, int N_nu = N_mu>
[[nodiscard]]
Matrix_3x3 CalculateCloverComponent(const GaugeField& U, const site_coord& current_site, const int mu, const int nu) noexcept
{
    static_assert(N_mu != 0 and N_nu != 0, "The template parameters of CalculateCloverComponent are not allowed to be 0!");
    if (mu == nu)
    {
        return Matrix_3x3::Zero();
    }
    else
    {
        return RectangularLoop<N_mu, N_nu>(U, current_site, mu, nu) + RectangularLoop<N_nu, -N_mu>(U, current_site, nu, mu) + RectangularLoop<-N_mu, -N_nu>(U, current_site, mu, nu) + RectangularLoop<-N_nu, N_mu>(U, current_site, nu, mu);
    }
}

// Template specialization for plaquette-based clover term
template<>
[[nodiscard]]
Matrix_3x3 CalculateCloverComponent<1, 1>(const GaugeField& U, const site_coord& current_site, const int mu, const int nu) noexcept
{
    if (mu == nu)
    {
        return Matrix_3x3::Zero();
    }
    else
    {
        return PlaquetteI(U, current_site, mu, nu) + PlaquetteII(U, current_site, mu, nu) + PlaquetteIII(U, current_site, mu, nu) + PlaquetteIV(U, current_site, mu, nu);
    }
}

template<int N_mu, int N_nu = N_mu>
void CalculateClover(const GaugeField& U, ZeroDiagonalAdjointSymmetricField& Clover) noexcept
{
    static_assert(N_mu != 0 and N_nu != 0, "The template parameters of CalculateClover are not allowed to be 0!");
    #pragma omp parallel for collapse(omp_collapse_depth)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        const site_coord current_site {t, x, y, z};
        for (int mu = 0;      mu < Ndim; ++mu)
        for (int nu = mu + 1; nu < Ndim; ++nu)
        {
            Clover.IndependentComponent(current_site, mu, nu) = CalculateCloverComponent<N_mu, N_nu>(U, current_site, mu, nu);
        }
    }
}

template<int N_mu, int N_nu = N_mu>
void CalculateCloverDifference(const GaugeField& U, AntisymmetricField& CloverDifference) noexcept
{
    static_assert(N_mu != 0 and N_nu != 0, "The template parameters of CalculateCloverDifference are not allowed to be 0!");
    #pragma omp parallel for collapse(omp_collapse_depth)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        const site_coord current_site {t, x, y, z};
        for (int mu = 0;      mu < Ndim; ++mu)
        for (int nu = mu + 1; nu < Ndim; ++nu)
        {
            const Matrix_3x3 component {CalculateCloverComponent<N_mu, N_nu>(U, current_site, mu, nu)};
            CloverDifference.IndependentComponent(current_site, mu, nu) = component - component.adjoint();
        }
    }
}

namespace Detail
{
    // Directly stores clover term
    inline void WriteFromCloverComponent(ZeroDiagonalAdjointSymmetricField& Clover, const site_coord& current_site, const int mu, const int nu, const Matrix_3x3& component) noexcept
    {
        Clover.IndependentComponent(current_site, mu, nu) = component;
    }

    // Stores antisymmetric clover difference term
    inline void WriteFromCloverComponent(AntisymmetricField& CloverDifference, const site_coord& current_site, const int mu, const int nu, const Matrix_3x3& component) noexcept
    {
        CloverDifference.IndependentComponent(current_site, mu, nu) = component - component.adjoint();
    }

    // Stores either the clover term or the clover difference term, depending on OutputFieldT
    template<typename OutputFieldT>
    void CalculatePlaquetteCloverComponents(const GaugeField& U, OutputFieldT& Output) noexcept
    {
        #pragma omp parallel for collapse(omp_collapse_depth)
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        {
            const int tm {(t - 1 + Nt)%Nt};
            const int xm {(x - 1 + Nx)%Nx};
            const int ym {(y - 1 + Ny)%Ny};
            const int zm {(z - 1 + Nz)%Nz};
            const int tp {(t + 1)%Nt};
            const int xp {(x + 1)%Nx};
            const int yp {(y + 1)%Ny};
            const int zp {(z + 1)%Nz};
            const site_coord current_site {t, x, y, z};

            const Matrix_3x3 component_01 {U({t, x, y, z, 0})            * U({tp, x, y, z, 1})            * U({t, xp, y, z, 0}).adjoint() * U({t, x, y, z, 1}).adjoint()
                                         + U({t, x, y, z, 1})            * U({tm, xp, y, z, 0}).adjoint() * U({tm, x, y, z, 1}).adjoint() * U({tm, x, y, z, 0})
                                         + U({tm, x, y, z, 0}).adjoint() * U({tm, xm, y, z, 1}).adjoint() * U({tm, xm, y, z, 0})          * U({t, xm, y, z, 1})
                                         + U({t, xm, y, z, 1}).adjoint() * U({t, xm, y, z, 0})            * U({tp, xm, y, z, 1})          * U({t, x, y, z, 0}).adjoint()};
            WriteFromCloverComponent(Output, current_site, 0, 1, component_01);

            const Matrix_3x3 component_02 {U({t, x, y, z, 0})            * U({tp, x, y, z, 2})            * U({t, x, yp, z, 0}).adjoint() * U({t, x, y, z, 2}).adjoint()
                                         + U({t, x, y, z, 2})            * U({tm, x, yp, z, 0}).adjoint() * U({tm, x, y, z, 2}).adjoint() * U({tm, x, y, z, 0})
                                         + U({tm, x, y, z, 0}).adjoint() * U({tm, x, ym, z, 2}).adjoint() * U({tm, x, ym, z, 0})          * U({t, x, ym, z, 2})
                                         + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, z, 0})            * U({tp, x, ym, z, 2})          * U({t, x, y, z, 0}).adjoint()};
            WriteFromCloverComponent(Output, current_site, 0, 2, component_02);

            const Matrix_3x3 component_03 {U({t, x, y, z, 0})            * U({tp, x, y, z, 3})            * U({t, x, y, zp, 0}).adjoint() * U({t, x, y, z, 3}).adjoint()
                                         + U({t, x, y, z, 3})            * U({tm, x, y, zp, 0}).adjoint() * U({tm, x, y, z, 3}).adjoint() * U({tm, x, y, z, 0})
                                         + U({tm, x, y, z, 0}).adjoint() * U({tm, x, y, zm, 3}).adjoint() * U({tm, x, y, zm, 0})          * U({t, x, y, zm, 3})
                                         + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 0})            * U({tp, x, y, zm, 3})          * U({t, x, y, z, 0}).adjoint()};
            WriteFromCloverComponent(Output, current_site, 0, 3, component_03);

            const Matrix_3x3 component_12 {U({t, x, y, z, 1})            * U({t, xp, y, z, 2})            * U({t, x, yp, z, 1}).adjoint() * U({t, x, y, z, 2}).adjoint()
                                         + U({t, x, y, z, 2})            * U({t, xm, yp, z, 1}).adjoint() * U({t, xm, y, z, 2}).adjoint() * U({t, xm, y, z, 1})
                                         + U({t, xm, y, z, 1}).adjoint() * U({t, xm, ym, z, 2}).adjoint() * U({t, xm, ym, z, 1})          * U({t, x, ym, z, 2})
                                         + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, z, 1})            * U({t, xp, ym, z, 2})          * U({t, x, y, z, 1}).adjoint()};
            WriteFromCloverComponent(Output, current_site, 1, 2, component_12);

            const Matrix_3x3 component_13 {U({t, x, y, z, 1})            * U({t, xp, y, z, 3})            * U({t, x, y, zp, 1}).adjoint() * U({t, x, y, z, 3}).adjoint()
                                         + U({t, x, y, z, 3})            * U({t, xm, y, zp, 1}).adjoint() * U({t, xm, y, z, 3}).adjoint() * U({t, xm, y, z, 1})
                                         + U({t, xm, y, z, 1}).adjoint() * U({t, xm, y, zm, 3}).adjoint() * U({t, xm, y, zm, 1})          * U({t, x, y, zm, 3})
                                         + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 1})            * U({t, xp, y, zm, 3})          * U({t, x, y, z, 1}).adjoint()};
            WriteFromCloverComponent(Output, current_site, 1, 3, component_13);

            const Matrix_3x3 component_23 {U({t, x, y, z, 2})            * U({t, x, yp, z, 3})            * U({t, x, y, zp, 2}).adjoint() * U({t, x, y, z, 3}).adjoint()
                                         + U({t, x, y, z, 3})            * U({t, x, ym, zp, 2}).adjoint() * U({t, x, ym, z, 3}).adjoint() * U({t, x, ym, z, 2})
                                         + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, zm, 3}).adjoint() * U({t, x, ym, zm, 2})          * U({t, x, y, zm, 3})
                                         + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 2})            * U({t, x, yp, zm, 3})          * U({t, x, y, z, 2}).adjoint()};
            WriteFromCloverComponent(Output, current_site, 2, 3, component_23);
        }

    }
} // namespace Detail

// Template specialization for plaquette-based clover term
template<>
void CalculateClover<1, 1>(const GaugeField& U, ZeroDiagonalAdjointSymmetricField& Clover) noexcept
{
    Detail::CalculatePlaquetteCloverComponents(U, Clover);
}

template<>
void CalculateCloverDifference<1, 1>(const GaugeField& U, AntisymmetricField& CloverDifference) noexcept
{
    Detail::CalculatePlaquetteCloverComponents(U, CloverDifference);
}

// // TODO: How do we generalize the function for abritrary clover sizes?
// [[nodiscard]]
// Matrix_3x3 CloverDerivativeComponent(const GaugeField& U, const FullTensor& Clover, const site_coord& current_site, const int mu, const int nu, const int rho, const int sigma) noexcept
// {
//     site_coord site_mup     = U.Move< 1>(current_site, mu);
//     site_coord site_nup     = U.Move< 1>(current_site, nu);
//     site_coord site_nud     = U.Move<-1>(current_site, nu);
//     site_coord site_mup_nup = U.Move< 1>(site_mup, nu);
//     site_coord site_mup_nud = U.Move<-1>(site_mup, nu);

//     // return U(current_site, mu) * (U(site_mup, nu)               * U(site_nup, mu).adjoint()        * U(current_site, nu).adjoint() * Clover(current_site, rho, sigma)
//     //                             + U(site_mup, nu)               * U(site_nup, mu).adjoint()        * Clover(site_nup, rho, sigma)  * U(current_site, nu).adjoint()
//     //                             + U(site_mup, nu)               * Clover(site_mup_nup, rho, sigma) * U(site_nup, mu).adjoint()     * U(current_site, nu).adjoint()
//     //                             + Clover(site_mup, rho, sigma)  * U(site_mup, nu)                  * U(site_nup, mu).adjoint()     * U(current_site, nu).adjoint()
//     //                             - U(site_mup_nud, nu).adjoint() * U(site_nud, mu).adjoint()        * U(site_nud, nu)               * Clover(current_site, rho, sigma)
//     //                             - U(site_mup_nud, nu).adjoint() * U(site_nud, mu).adjoint()        * Clover(site_nud, rho, sigma)  * U(site_nud, nu)
//     //                             - U(site_mup_nud, nu).adjoint() * Clover(site_mup_nud, rho, sigma) * U(site_nud, mu).adjoint()     * U(site_nud, nu)
//     //                             - Clover(site_mup, rho, sigma)  * U(site_mup_nud, nu).adjoint()    * U(site_nud, mu).adjoint()     * U(site_nud, nu));
//     // We can move the first multiplication outside into the function CloverDerivative(), would go from 6 -> 1 multiplications (although relatively speaking it's not that much)
//     // TODO: Factor out more common products? Might slightly improve performance at the cost of readability
//     return (U(site_mup, nu)               * U(site_nup, mu).adjoint()        * U(current_site, nu).adjoint() * Clover(current_site, rho, sigma)
//           + U(site_mup, nu)               * U(site_nup, mu).adjoint()        * Clover(site_nup, rho, sigma)  * U(current_site, nu).adjoint()
//           + U(site_mup, nu)               * Clover(site_mup_nup, rho, sigma) * U(site_nup, mu).adjoint()     * U(current_site, nu).adjoint()
//           + Clover(site_mup, rho, sigma)  * U(site_mup, nu)                  * U(site_nup, mu).adjoint()     * U(current_site, nu).adjoint()
//           - U(site_mup_nud, nu).adjoint() * U(site_nud, mu).adjoint()        * U(site_nud, nu)               * Clover(current_site, rho, sigma)
//           - U(site_mup_nud, nu).adjoint() * U(site_nud, mu).adjoint()        * Clover(site_nud, rho, sigma)  * U(site_nud, nu)
//           - U(site_mup_nud, nu).adjoint() * Clover(site_mup_nud, rho, sigma) * U(site_nud, mu).adjoint()     * U(site_nud, nu)
//           - Clover(site_mup, rho, sigma)  * U(site_mup_nud, nu).adjoint()    * U(site_nud, mu).adjoint()     * U(site_nud, nu));
// }

// // TODO: How do we generalize the function for abritrary clover sizes?
// [[nodiscard]]
// Matrix_3x3 CloverDerivative(const GaugeField& U, const FullTensor& Clover, const site_coord& current_site, const int mu) noexcept
// {
//     Matrix_3x3 derivative_component {Matrix_3x3::Zero()};
//     // This is basically epsilon_{mu, nu, rho, sigma} manually worked out (not sure if writing a function makes sense?)
//     switch (mu)
//     {
//         case 0:
//         {
//             derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 1, 2, 3);
//             derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 1, 3, 2);
//             derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 2, 1, 3);
//             derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 2, 3, 1);
//             derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 3, 1, 2);
//             derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 3, 2, 1);
//         }
//         break;
//         case 1:
//         {
//             derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 0, 2, 3);
//             derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 0, 3, 2);
//             derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 2, 0, 3);
//             derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 2, 3, 0);
//             derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 3, 0, 2);
//             derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 3, 2, 0);
//         }
//         break;
//         case 2:
//         {
//             derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 0, 1, 3);
//             derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 0, 3, 1);
//             derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 1, 0, 3);
//             derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 1, 3, 0);
//             derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 3, 0, 1);
//             derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 3, 1, 0);
//         }
//         break;
//         case 3:
//         {
//             derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 0, 1, 2);
//             derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 0, 2, 1);
//             derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 1, 0, 2);
//             derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 1, 2, 0);
//             derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 2, 0, 1);
//             derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 2, 1, 0);
//         }
//         break;
//     }
//     // The link U(current_site, mu) is a common factor appearing in all 6 components of the clover derivative
//     derivative_component = U(current_site, mu) * derivative_component;
//     return -static_cast<floatT>(1.0/(256.0 * pi<floatT> * pi<floatT>)) * SU3::Projection::Algebra(derivative_component);
// }

[[nodiscard]]
Matrix_3x3 CloverDerivativeContribution(const GaugeField& U, const AntisymmetricField& CloverDifference, const site_coord& current_site, const site_coord& site_mup, const int mu, const int nu, const int rho, const int sigma) noexcept
{
    const site_coord site_nup     {U.Move< 1>(current_site, nu)};
    const site_coord site_nud     {U.Move<-1>(current_site, nu)};
    const site_coord site_mup_nup {U.Move< 1>(site_mup, nu)};
    const site_coord site_mup_nud {U.Move<-1>(site_mup, nu)};

    // We can replace all clover terms C_{rho sigma} with R_{rho sigma} = C_{rho sigma} - C_{sigma rho}, which means we only need half of the clover terms derivative components (due to linearity and antisymmetry of R)
    const Matrix_3x3& R_current_site {CloverDifference.IndependentComponent(current_site, rho, sigma)};
    const Matrix_3x3& R_site_nup     {CloverDifference.IndependentComponent(    site_nup, rho, sigma)};
    const Matrix_3x3& R_site_mup_nup {CloverDifference.IndependentComponent(site_mup_nup, rho, sigma)};
    const Matrix_3x3& R_site_mup     {CloverDifference.IndependentComponent(    site_mup, rho, sigma)};
    const Matrix_3x3& R_site_nud     {CloverDifference.IndependentComponent(    site_nud, rho, sigma)};
    const Matrix_3x3& R_site_mup_nud {CloverDifference.IndependentComponent(site_mup_nud, rho, sigma)};

    // Original return statement without reusing common factors (left here due to better readability)
    // return (U(site_mup, nu)               * U(site_nup, mu).adjoint()     * U(current_site, nu).adjoint() * R_current_site
    //       + U(site_mup, nu)               * U(site_nup, mu).adjoint()     * R_site_nup                    * U(current_site, nu).adjoint()
    //       + U(site_mup, nu)               * R_site_mup_nup                * U(site_nup, mu).adjoint()     * U(current_site, nu).adjoint()
    //       + R_site_mup                    * U(site_mup, nu)               * U(site_nup, mu).adjoint()     * U(current_site, nu).adjoint()
    //       - U(site_mup_nud, nu).adjoint() * U(site_nud, mu).adjoint()     * U(site_nud, nu)               * R_current_site
    //       - U(site_mup_nud, nu).adjoint() * U(site_nud, mu).adjoint()     * R_site_nud                    * U(site_nud, nu)
    //       - U(site_mup_nud, nu).adjoint() * R_site_mup_nud                * U(site_nud, mu).adjoint()     * U(site_nud, nu)
    //       - R_site_mup                    * U(site_mup_nud, nu).adjoint() * U(site_nud, mu).adjoint()     * U(site_nud, nu));

    // Reuse terms occuring more than once
    const Matrix_SU3 U_current_nu_adj {U(current_site, nu).adjoint()};
    const Matrix_SU3 U_mup_nud_nu_adj {U(site_mup_nud, nu).adjoint()};
    const Matrix_SU3 tmp1             {U(    site_mup, nu)           * U(    site_nup, mu).adjoint()};
    const Matrix_SU3 tmp2             {U(    site_nup, mu).adjoint() * U_current_nu_adj             };
    const Matrix_SU3 tmp3             {U_mup_nud_nu_adj              * U(    site_nud, mu).adjoint()};
    const Matrix_SU3 tmp4             {U(    site_nud, mu).adjoint() * U(    site_nud, nu)          };

    return (tmp1 * ( U_current_nu_adj * R_current_site + R_site_nup * U_current_nu_adj )
          +        ( U(site_mup, nu)  * R_site_mup_nup + R_site_mup * U(site_mup, nu)  ) * tmp2
          - tmp3 * ( U(site_nud, nu)  * R_current_site + R_site_nud * U(site_nud, nu)  )
          -        ( U_mup_nud_nu_adj * R_site_mup_nud + R_site_mup * U_mup_nud_nu_adj ) * tmp4);
}

// TODO: How do we generalize the function for abritrary clover sizes?
[[nodiscard]]
Matrix_3x3 CloverDerivative(const GaugeField& U, const AntisymmetricField& CloverDifference, const site_coord& current_site, const int mu) noexcept
{
    Matrix_3x3       derivative_component {Matrix_3x3::Zero()};
    const site_coord site_mup             {U.Move<1>(current_site, mu)};
    // Only explicitly evaluate non-zero contributions (epsilon_{mu nu rho sigma} = +/- 1)
    // Each direction can be reduced to three independent terms due to the antisymmetry of R_{rho sigma} = C_{rho sigma} - C_{sigma rho}
    // The additional factor 2 is included below in the return statement
    switch (mu)
    {
        case 0:
        {
            derivative_component += CloverDerivativeContribution(U, CloverDifference, current_site, site_mup, mu, 1, 2, 3);
            derivative_component -= CloverDerivativeContribution(U, CloverDifference, current_site, site_mup, mu, 2, 1, 3);
            derivative_component += CloverDerivativeContribution(U, CloverDifference, current_site, site_mup, mu, 3, 1, 2);
        }
        break;
        case 1:
        {
            derivative_component -= CloverDerivativeContribution(U, CloverDifference, current_site, site_mup, mu, 0, 2, 3);
            derivative_component += CloverDerivativeContribution(U, CloverDifference, current_site, site_mup, mu, 2, 0, 3);
            derivative_component -= CloverDerivativeContribution(U, CloverDifference, current_site, site_mup, mu, 3, 0, 2);
        }
        break;
        case 2:
        {
            derivative_component += CloverDerivativeContribution(U, CloverDifference, current_site, site_mup, mu, 0, 1, 3);
            derivative_component -= CloverDerivativeContribution(U, CloverDifference, current_site, site_mup, mu, 1, 0, 3);
            derivative_component += CloverDerivativeContribution(U, CloverDifference, current_site, site_mup, mu, 3, 0, 1);
        }
        break;
        case 3:
        {
            derivative_component -= CloverDerivativeContribution(U, CloverDifference, current_site, site_mup, mu, 0, 1, 2);
            derivative_component += CloverDerivativeContribution(U, CloverDifference, current_site, site_mup, mu, 1, 0, 2);
            derivative_component -= CloverDerivativeContribution(U, CloverDifference, current_site, site_mup, mu, 2, 0, 1);
        }
        break;
    }
    // The link U(current_site, mu) is a common factor appearing in all 6 components of the clover derivative
    // TODO: Prefactor for different Ncolor or Ndim?
    derivative_component = U(current_site, mu) * derivative_component;
    return -static_cast<floatT>(2.0/(512.0 * pi<floatT> * pi<floatT>)) * SU3::Projection::Algebra(derivative_component);
}

#endif // LETTUCE_CLOVER_HPP
