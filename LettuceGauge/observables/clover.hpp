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
void CalculateClover(const GaugeField& U, FullTensor& Clov) noexcept
{
    static_assert(N_mu != 0 and N_nu != 0, "The template parameters of CalculateClover are not allowed to be 0!");
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        site_coord current_site {t, x, y, z};
        Clov(current_site, 0, 0).setZero();
        Clov(current_site, 0, 1) = CalculateCloverComponent<N_mu, N_nu>(U, current_site, 0, 1);
        Clov(current_site, 1, 0) = Clov(current_site, 0, 1).adjoint();

        Clov(current_site, 0, 2) = CalculateCloverComponent<N_mu, N_nu>(U, current_site, 0, 2);
        Clov(current_site, 2, 0) = Clov(current_site, 0, 2).adjoint();

        Clov(current_site, 0, 3) = CalculateCloverComponent<N_mu, N_nu>(U, current_site, 0, 3);
        Clov(current_site, 3, 0) = Clov(current_site, 0, 3).adjoint();

        Clov(current_site, 1, 1).setZero();
        Clov(current_site, 1, 2) = CalculateCloverComponent<N_mu, N_nu>(U, current_site, 1, 2);
        Clov(current_site, 2, 1) = Clov(current_site, 1, 2).adjoint();

        Clov(current_site, 1, 3) = CalculateCloverComponent<N_mu, N_nu>(U, current_site, 1, 3);
        Clov(current_site, 3, 1) = Clov(current_site, 1, 3).adjoint();

        Clov(current_site, 2, 2).setZero();
        Clov(current_site, 2, 3) = CalculateCloverComponent<N_mu, N_nu>(U, current_site, 2, 3);
        Clov(current_site, 3, 2) = Clov(current_site, 2, 3).adjoint();
        Clov(current_site, 3, 3).setZero();
    }
}

// Template specialization for plaquette-based clover term
template<>
void CalculateClover<1, 1>(const GaugeField& U, FullTensor& Clov) noexcept
{
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        int tm = (t - 1 + Nt)%Nt;
        int xm = (x - 1 + Nx)%Nx;
        int ym = (y - 1 + Ny)%Ny;
        int zm = (z - 1 + Nz)%Nz;
        int tp = (t + 1)%Nt;
        int xp = (x + 1)%Nx;
        int yp = (y + 1)%Ny;
        int zp = (z + 1)%Nz;

        Clov(t, x, y, z, 0, 0).setZero();
        Clov(t, x, y, z, 0, 1) = U({t, x, y, z, 0})            * U({tp, x, y, z, 1})            * U({t, xp, y, z, 0}).adjoint() * U({t, x, y, z, 1}).adjoint()
                               + U({t, x, y, z, 1})            * U({tm, xp, y, z, 0}).adjoint() * U({tm, x, y, z, 1}).adjoint() * U({tm, x, y, z, 0})
                               + U({tm, x, y, z, 0}).adjoint() * U({tm, xm, y, z, 1}).adjoint() * U({tm, xm, y, z, 0})          * U({t, xm, y, z, 1})
                               + U({t, xm, y, z, 1}).adjoint() * U({t, xm, y, z, 0})            * U({tp, xm, y, z, 1})          * U({t, x, y, z, 0}).adjoint();
        Clov(t, x, y, z, 1, 0) = Clov(t, x, y, z, 0, 1).adjoint();

        Clov(t, x, y, z, 0, 2) = U({t, x, y, z, 0})            * U({tp, x, y, z, 2})            * U({t, x, yp, z, 0}).adjoint() * U({t, x, y, z, 2}).adjoint()
                               + U({t, x, y, z, 2})            * U({tm, x, yp, z, 0}).adjoint() * U({tm, x, y, z, 2}).adjoint() * U({tm, x, y, z, 0})
                               + U({tm, x, y, z, 0}).adjoint() * U({tm, x, ym, z, 2}).adjoint() * U({tm, x, ym, z, 0})          * U({t, x, ym, z, 2})
                               + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, z, 0})            * U({tp, x, ym, z, 2})          * U({t, x, y, z, 0}).adjoint();
        Clov(t, x, y, z, 2, 0) = Clov(t, x, y, z, 0, 2).adjoint();

        Clov(t, x, y, z, 0, 3) = U({t, x, y, z, 0})            * U({tp, x, y, z, 3})            * U({t, x, y, zp, 0}).adjoint() * U({t, x, y, z, 3}).adjoint()
                               + U({t, x, y, z, 3})            * U({tm, x, y, zp, 0}).adjoint() * U({tm, x, y, z, 3}).adjoint() * U({tm, x, y, z, 0})
                               + U({tm, x, y, z, 0}).adjoint() * U({tm, x, y, zm, 3}).adjoint() * U({tm, x, y, zm, 0})          * U({t, x, y, zm, 3})
                               + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 0})            * U({tp, x, y, zm, 3})          * U({t, x, y, z, 0}).adjoint();
        Clov(t, x, y, z, 3, 0) = Clov(t, x, y, z, 0, 3).adjoint();

        Clov(t, x, y, z, 1, 1).setZero();
        Clov(t, x, y, z, 1, 2) = U({t, x, y, z, 1})            * U({t, xp, y, z, 2})            * U({t, x, yp, z, 1}).adjoint() * U({t, x, y, z, 2}).adjoint()
                               + U({t, x, y, z, 2})            * U({t, xm, yp, z, 1}).adjoint() * U({t, xm, y, z, 2}).adjoint() * U({t, xm, y, z, 1})
                               + U({t, xm, y, z, 1}).adjoint() * U({t, xm, ym, z, 2}).adjoint() * U({t, xm, ym, z, 1})          * U({t, x, ym, z, 2})
                               + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, z, 1})            * U({t, xp, ym, z, 2})          * U({t, x, y, z, 1}).adjoint();
        Clov(t, x, y, z, 2, 1) = Clov(t, x, y, z, 1, 2).adjoint();

        Clov(t, x, y, z, 1, 3) = U({t, x, y, z, 1})            * U({t, xp, y, z, 3})            * U({t, x, y, zp, 1}).adjoint() * U({t, x, y, z, 3}).adjoint()
                               + U({t, x, y, z, 3})            * U({t, xm, y, zp, 1}).adjoint() * U({t, xm, y, z, 3}).adjoint() * U({t, xm, y, z, 1})
                               + U({t, xm, y, z, 1}).adjoint() * U({t, xm, y, zm, 3}).adjoint() * U({t, xm, y, zm, 1})          * U({t, x, y, zm, 3})
                               + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 1})            * U({t, xp, y, zm, 3})          * U({t, x, y, z, 1}).adjoint();
        Clov(t, x, y, z, 3, 1) = Clov(t, x, y, z, 1, 3).adjoint();

        Clov(t, x, y, z, 2, 2).setZero();
        Clov(t, x, y, z, 2, 3) = U({t, x, y, z, 2})            * U({t, x, yp, z, 3})            * U({t, x, y, zp, 2}).adjoint() * U({t, x, y, z, 3}).adjoint()
                               + U({t, x, y, z, 3})            * U({t, x, ym, zp, 2}).adjoint() * U({t, x, ym, z, 3}).adjoint() * U({t, x, ym, z, 2})
                               + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, zm, 3}).adjoint() * U({t, x, ym, zm, 2})          * U({t, x, y, zm, 3})
                               + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 2})            * U({t, x, yp, zm, 3})          * U({t, x, y, z, 2}).adjoint();
        Clov(t, x, y, z, 3, 2) = Clov(t, x, y, z, 2, 3).adjoint();
        Clov(t, x, y, z, 3, 3).setZero();
    }
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
Matrix_3x3 CloverDerivativeComponent(const GaugeField& U, const FullTensor& Clover, const site_coord& current_site, const int mu, const int nu, const int rho, const int sigma) noexcept
{
    site_coord site_mup     = U.Move< 1>(current_site, mu);
    site_coord site_nup     = U.Move< 1>(current_site, nu);
    site_coord site_nud     = U.Move<-1>(current_site, nu);
    site_coord site_mup_nup = U.Move< 1>(site_mup, nu);
    site_coord site_mup_nud = U.Move<-1>(site_mup, nu);

    // We can move the first multiplication outside into the function CloverDerivative(), would go from 6 -> 1 multiplications (although relatively speaking it's not that much)
    // We can also replace all clover terms C_{rho, sigma} with R_{rho, sigma} = C_{rho, sigma} - C_{sigma, rho}, which means we only need half of the clover terms derivative components (due to linearity and antisymmetry of R)
    Matrix_3x3 R_current_site {Clover(current_site, rho, sigma) - Clover(current_site, sigma, rho)};
    Matrix_3x3 R_site_nup     {Clover(    site_nup, rho, sigma) - Clover(    site_nup, sigma, rho)};
    Matrix_3x3 R_site_mup_nup {Clover(site_mup_nup, rho, sigma) - Clover(site_mup_nup, sigma, rho)};
    Matrix_3x3 R_site_mup     {Clover(    site_mup, rho, sigma) - Clover(    site_mup, sigma, rho)};
    Matrix_3x3 R_site_nud     {Clover(    site_nud, rho, sigma) - Clover(    site_nud, sigma, rho)};
    Matrix_3x3 R_site_mup_nud {Clover(site_mup_nud, rho, sigma) - Clover(site_mup_nud, sigma, rho)};
    return (U(site_mup, nu)               * U(site_nup, mu).adjoint()     * U(current_site, nu).adjoint() * R_current_site
          + U(site_mup, nu)               * U(site_nup, mu).adjoint()     * R_site_nup                    * U(current_site, nu).adjoint()
          + U(site_mup, nu)               * R_site_mup_nup                * U(site_nup, mu).adjoint()     * U(current_site, nu).adjoint()
          + R_site_mup                    * U(site_mup, nu)               * U(site_nup, mu).adjoint()     * U(current_site, nu).adjoint()
          - U(site_mup_nud, nu).adjoint() * U(site_nud, mu).adjoint()     * U(site_nud, nu)               * R_current_site
          - U(site_mup_nud, nu).adjoint() * U(site_nud, mu).adjoint()     * R_site_nud                    * U(site_nud, nu)
          - U(site_mup_nud, nu).adjoint() * R_site_mup_nud                * U(site_nud, mu).adjoint()     * U(site_nud, nu)
          - R_site_mup                    * U(site_mup_nud, nu).adjoint() * U(site_nud, mu).adjoint()     * U(site_nud, nu));
}

// TODO: How do we generalize the function for abritrary clover sizes?
[[nodiscard]]
Matrix_3x3 CloverDerivative(const GaugeField& U, const FullTensor& Clover, const site_coord& current_site, const int mu) noexcept
{
    Matrix_3x3 derivative_component {Matrix_3x3::Zero()};
    // This is basically epsilon_{mu, nu, rho, sigma} manually worked out (not sure if writing a function makes sense?)
    switch (mu)
    {
        // Only three terms per direction are needed here since we can use the antisymmetry of R_{rho, sigma} = C_{rho, sigma} - C_{sigma, rho}
        // The additional factor 2 is included below in the return statement
        case 0:
        {
            derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 1, 2, 3);
            derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 2, 1, 3);
            derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 3, 1, 2);
        }
        break;
        case 1:
        {
            derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 0, 2, 3);
            derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 2, 0, 3);
            derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 3, 0, 2);
        }
        break;
        case 2:
        {
            derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 0, 1, 3);
            derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 1, 0, 3);
            derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 3, 0, 1);
        }
        break;
        case 3:
        {
            derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 0, 1, 2);
            derivative_component += CloverDerivativeComponent(U, Clover, current_site, mu, 1, 0, 2);
            derivative_component -= CloverDerivativeComponent(U, Clover, current_site, mu, 2, 0, 1);
        }
        break;
    }
    // The link U(current_site, mu) is a common factor appearing in all 6 components of the clover derivative, so multiply here
    // TODO: Currently not sure how to generalize the prefactor to SU(Ncolor) in Ndim dimensions (or if necessary at all)
    derivative_component = U(current_site, mu) * derivative_component;
    return -static_cast<floatT>(2.0/(512.0 * pi<floatT> * pi<floatT>)) * SU3::Projection::Algebra(derivative_component);
}

#endif // LETTUCE_CLOVER_HPP
