#ifndef LETTUCE_WILSON_LOOP_HPP
#define LETTUCE_WILSON_LOOP_HPP

// Non-standard library headers
#include "../coords.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <complex>
//----------------------------------------
// Standard C headers
// ...

//+---------------------------------------------------------------------------------+
//| This file provides several function templates that allow the calculation of     |
//| single square and rectangular Wilson loops.                                     |
//+---------------------------------------------------------------------------------+


// TODO: Perhaps this should be moved out of wilson_loop.hpp, as the function can be used much more generically
// Computes the product of N_mu links along a straight path in direction mu (a negative N_mu corresponds to moving backwards), starting from current_site
template<int N_mu>
[[nodiscard]]
Matrix_3x3 LineProduct(const GaugeField& U, const site_coord& current_site, const int mu) noexcept
{
    static_assert(N_mu != 0, "The template parameter of LineProduct is not allowed to be 0!");
    // Check sign of N_mu to see if we go forwards or backwards
    constexpr int          sign_mu  {(N_mu > 0) - (N_mu < 0)};
    // As of C++20, std::abs is not yet constexpr (coming in C++23), so for now multiply with sign_mu (should be okay since we asserted N_mu != 0)
    constexpr unsigned int dist_mu  {sign_mu * N_mu};
    if constexpr(sign_mu > 0)
    {
        site_coord tmp_site     {current_site};
        Matrix_3x3 link_product {U(tmp_site, mu)};
        // Since we already initialized link_product as U(current_site, mu), the loop starts from 1
        for (int mu_count = 1; mu_count < dist_mu; ++mu_count)
        {
            tmp_site = Move<1>(tmp_site, mu);
            link_product *= U(tmp_site, mu);
        }
        return link_product;
    }
    else
    {
        site_coord tmp_site     {Move<-1>(current_site, mu)};
        Matrix_3x3 link_product {U(tmp_site, mu).adjoint()};
        // Since we already initialized link_product as U(current_site - mu, mu), the loop starts from 1
        for (int mu_count = 1; mu_count < dist_mu; ++mu_count)
        {
            tmp_site = Move<-1>(tmp_site, mu);
            link_product *= U(tmp_site, mu).adjoint();
        }
        return link_product;
    }
}

// TODO: Does it make sense to precompute U_chain, or is it more efficient to locally calculate the terms?
//       For instance, we could left multiply with the adjoint/inverse and right multiply with a new link, which
//       might be computationally advantageous for larger chain lengths (only 2 multiplications instead of N)
template<int N_mu_start, int N_mu_end, bool reset>
[[nodiscard]]
double WilsonLoop(const GaugeField& U, GaugeField& U_chain) noexcept
{
    static_assert(N_mu_start >= 0 and N_mu_end >= 0, "The template parameters of WilsonLoop are not allowed to be negative!");
    double W {0.0};
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        if constexpr(reset)
        {
            for (int mu = 0; mu < 4; ++mu)
            {
                U_chain({t, x, y, z, mu}).setIdentity();
            }
        }
        for (int n = N_mu_start; n < N_mu_end; ++n)
        {
            U_chain({t, x, y, z, 0}) *= U({(t + n)%Nt, x, y, z, 0});
            U_chain({t, x, y, z, 1}) *= U({t, (x + n)%Nx, y, z, 1});
            U_chain({t, x, y, z, 2}) *= U({t, x, (y + n)%Ny, z, 2});
            U_chain({t, x, y, z, 3}) *= U({t, x, y, (z + n)%Nz, 3});
        }
    }
    #pragma omp parallel for reduction(+: W)
    for (int t = 0; t < Nt; ++t)
    {
        int tp {(t + N_mu_end)%Nt};
        for (int x = 0; x < Nx; ++x)
        {
            int xp {(x + N_mu_end)%Nx};
            for (int y = 0; y < Ny; ++y)
            {
                int yp {(y + N_mu_end)%Ny};
                for (int z = 0; z < Nz; ++z)
                {
                    int zp {(z + N_mu_end)%Nz};
                    // W += (U_chain({t, x, y, z, 0}) * U_chain({tp, x, y, z, 1}) * U_chain({t, xp, y, z, 0}).adjoint() * U_chain({t, x, y, z, 1}).adjoint()).cast<std::complex<double>>();
                    // W += (U_chain({t, x, y, z, 0}) * U_chain({tp, x, y, z, 2}) * U_chain({t, x, yp, z, 0}).adjoint() * U_chain({t, x, y, z, 2}).adjoint()).cast<std::complex<double>>();
                    // W += (U_chain({t, x, y, z, 0}) * U_chain({tp, x, y, z, 3}) * U_chain({t, x, y, zp, 0}).adjoint() * U_chain({t, x, y, z, 3}).adjoint()).cast<std::complex<double>>();
                    // W += (U_chain({t, x, y, z, 1}) * U_chain({t, xp, y, z, 2}) * U_chain({t, x, yp, z, 1}).adjoint() * U_chain({t, x, y, z, 2}).adjoint()).cast<std::complex<double>>();
                    // W += (U_chain({t, x, y, z, 1}) * U_chain({t, xp, y, z, 3}) * U_chain({t, x, y, zp, 1}).adjoint() * U_chain({t, x, y, z, 3}).adjoint()).cast<std::complex<double>>();
                    // W += (U_chain({t, x, y, z, 2}) * U_chain({t, x, yp, z, 3}) * U_chain({t, x, y, zp, 2}).adjoint() * U_chain({t, x, y, z, 3}).adjoint()).cast<std::complex<double>>();

                    W += std::real((U_chain({t, x, y, z, 0}) * U_chain({tp, x, y, z, 1}) * U_chain({t, xp, y, z, 0}).adjoint() * U_chain({t, x, y, z, 1}).adjoint()).trace());
                    W += std::real((U_chain({t, x, y, z, 0}) * U_chain({tp, x, y, z, 2}) * U_chain({t, x, yp, z, 0}).adjoint() * U_chain({t, x, y, z, 2}).adjoint()).trace());
                    W += std::real((U_chain({t, x, y, z, 0}) * U_chain({tp, x, y, z, 3}) * U_chain({t, x, y, zp, 0}).adjoint() * U_chain({t, x, y, z, 3}).adjoint()).trace());
                    W += std::real((U_chain({t, x, y, z, 1}) * U_chain({t, xp, y, z, 2}) * U_chain({t, x, yp, z, 1}).adjoint() * U_chain({t, x, y, z, 2}).adjoint()).trace());
                    W += std::real((U_chain({t, x, y, z, 1}) * U_chain({t, xp, y, z, 3}) * U_chain({t, x, y, zp, 1}).adjoint() * U_chain({t, x, y, z, 3}).adjoint()).trace());
                    W += std::real((U_chain({t, x, y, z, 2}) * U_chain({t, x, yp, z, 3}) * U_chain({t, x, y, zp, 2}).adjoint() * U_chain({t, x, y, z, 3}).adjoint()).trace());

                    // W += std::real((U_chain({t1, x1, y1, z1, 0}) * U_chain({t2, x1, y1, z1, 1}) * U_chain({t1, x2, y1, z1, 0}).adjoint() * U_chain({t1, x1, y1, z1, 1}).adjoint()).trace());
                    // W += std::real((U_chain({t1, x1, y1, z1, 0}) * U_chain({t2, x1, y1, z1, 2}) * U_chain({t1, x1, y2, z1, 0}).adjoint() * U_chain({t1, x1, y1, z1, 2}).adjoint()).trace());
                    // W += std::real((U_chain({t1, x1, y1, z1, 0}) * U_chain({t2, x1, y1, z1, 3}) * U_chain({t1, x1, y1, z2, 0}).adjoint() * U_chain({t1, x1, y1, z1, 3}).adjoint()).trace());
                    // W += std::real((U_chain({t1, x1, y1, z1, 1}) * U_chain({t1, x2, y1, z1, 2}) * U_chain({t1, x1, y2, z1, 1}).adjoint() * U_chain({t1, x1, y1, z1, 2}).adjoint()).trace());
                    // W += std::real((U_chain({t1, x1, y1, z1, 1}) * U_chain({t1, x2, y1, z1, 3}) * U_chain({t1, x1, y1, z2, 1}).adjoint() * U_chain({t1, x1, y1, z1, 3}).adjoint()).trace());
                    // W += std::real((U_chain({t1, x1, y1, z1, 2}) * U_chain({t1, x1, y2, z1, 3}) * U_chain({t1, x1, y1, z2, 2}).adjoint() * U_chain({t1, x1, y1, z1, 3}).adjoint()).trace());
                }
            }
        }
    }
    return 1.0 - W/(18.0 * U.Volume());
}

// Function template to calculate a single (square) Wilson loop of arbitrary length
template<int N_mu, int N_nu>
Matrix_SU3 WilsonLoop(const GaugeField& U, const site_coord current_site, const int mu, const int nu) noexcept
{
    static_assert(N_mu != 0 and N_nu != 0, "The template parameters of WilsonLoop are not allowed to be 0!");
    static_assert((N_mu * N_mu) == (N_nu * N_nu), "The absolute values of the template parameters of WilsonLoop must be the same!");
    site_coord site_mup     {Move<N_mu>(current_site, mu)};
    site_coord site_mup_nup {Move<N_nu>(site_mup,     nu)};
    site_coord site_nup     {Move<N_nu>(current_site, nu)};
    return LineProduct<N_mu>(U, current_site, mu) * LineProduct<N_nu>(U, site_mup, nu) * LineProduct<-N_mu>(U, site_mup_nup, mu) * LineProduct<-N_nu>(U, site_nup, nu);
}

// A more general version of the function template above which allows for rectangular loops
template<int N_mu, int N_nu>
[[nodiscard]]
Matrix_SU3 RectangularLoop(const GaugeField& U, const site_coord& current_site, const int mu, const int nu) noexcept
{
    static_assert(N_mu != 0 and N_nu != 0, "The template parameters of RectangularLoop are not allowed to be 0!");
    site_coord site_mup     {Move<N_mu>(current_site, mu)};
    site_coord site_mup_nup {Move<N_nu>(site_mup,     nu)};
    site_coord site_nup     {Move<N_nu>(current_site, nu)};
    return LineProduct<N_mu>(U, current_site, mu) * LineProduct<N_nu>(U, site_mup, nu) * LineProduct<-N_mu>(U, site_mup_nup, mu) * LineProduct<-N_nu>(U, site_nup, nu);
}

// Template specializations for 1x2 and 2x1 loops used in rectangular gauge actions
template<>
[[nodiscard]]
Matrix_SU3 RectangularLoop<1, 2>(const GaugeField& U, const site_coord& current_site, const int mu, const int nu) noexcept
{
    site_coord site_mup     {Move<1>(current_site, mu)};
    site_coord site_mup_nup {Move<1>(site_mup    , nu)};
    site_coord site_nupp    {Move<2>(current_site, nu)};
    site_coord site_nup     {Move<1>(current_site, nu)};
    return U(current_site, mu) * U(site_mup, nu) * U(site_mup_nup, nu) * U(site_nupp, mu).adjoint() * U(site_nup, nu).adjoint() * U(current_site, nu).adjoint();
}

template<>
[[nodiscard]]
Matrix_SU3 RectangularLoop<2, 1>(const GaugeField& U, const site_coord& current_site, const int mu, const int nu) noexcept
{
    site_coord site_mup     {Move<1>(current_site, mu)};
    site_coord site_mupp    {Move<1>(site_mup    , mu)};
    site_coord site_mup_nup {Move<1>(site_mup    , nu)};
    site_coord site_nup     {Move<1>(current_site, nu)};
    return U(current_site, mu) * U(site_mup, mu) * U(site_mupp, nu) * U(site_mup_nup, mu).adjoint() * U(site_nup, mu).adjoint() * U(current_site, nu).adjoint();
}

#endif // LETTUCE_WILSON_LOOP_HPP
