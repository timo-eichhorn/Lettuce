#ifndef LETTUCE_PLAQUETTE_HPP
#define LETTUCE_PLAQUETTE_HPP

// Non-standard library headers
#include "../defines.hpp"
#include <Eigen/Dense>
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <complex>
//----------------------------------------
// Standard C headers
#include <cmath>

//-----
// Calculates plaquette at given coordinates

[[nodiscard]]
Matrix_SU3 Plaquette(const GaugeField& U, const site_coord& current_site, const int mu, const int nu) noexcept
{
    return U(current_site, mu) * U(U.Move<1>(current_site, mu), nu) * U(U.Move<1>(current_site, nu), mu).adjoint() * U(current_site, nu).adjoint();
}

[[nodiscard]]
double PlaquetteSum(const GaugeField& U) noexcept
{
    double Plaq_sum {0.0};
    #pragma omp parallel for collapse(2) reduction(+: Plaq_sum)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int nu = 1; nu < 4; ++nu)
    {
        for (int mu = 0; mu < nu; ++mu)
        {
            Plaq_sum += std::real(Plaquette(U, {t, x, y, z}, mu, nu).trace());
        }
    }
    return Plaq_sum;
}

[[nodiscard]]
double PlaquetteSumTimeslice(const GaugeField& U, const int t) noexcept
{
    double Plaq_sum {0.0};
    #pragma omp parallel for collapse(2) reduction(+: Plaq_sum)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int nu = 1; nu < 4; ++nu)
    {
        site_coord current_site {t, x, y, z};
        for (int mu = 0; mu < nu; ++mu)
        {
            Plaq_sum += std::real(Plaquette(U, current_site, mu, nu).trace());
        }
    }
    return Plaq_sum;
}

[[nodiscard]]
double MaxPlaquette(const GaugeField& U) noexcept
{
    double max_plaquette {-static_cast<double>(Ncolor)};
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int nu = 1; nu < 4; ++nu)
    {
        for (int mu = 0; mu < nu; ++mu)
        {
            max_plaquette = std::fmax(max_plaquette, std::real(Plaquette(U, {t, x, y, z}, mu, nu).trace()));
        }
    }
    return max_plaquette;
}

// Top right quadrant, i.e., P_{mu, nu}
[[nodiscard]]
Matrix_SU3 PlaquetteI(const GaugeField& U, const site_coord& current_site, const int mu, const int nu) noexcept
{
    return U(current_site, mu) * U(U.Move<1>(current_site, mu), nu) * U(U.Move<1>(current_site, nu), mu).adjoint() * U(current_site, nu).adjoint();
}

// Top left quadrant, i.e., P_{nu, -mu}
[[nodiscard]]
Matrix_SU3 PlaquetteII(const GaugeField& U, const site_coord& current_site, const int mu, const int nu) noexcept
{
    site_coord site_mud     {U.Move<-1>(current_site, mu)};
    return U(current_site, nu) * U(U.Move<1>(site_mud, nu), mu).adjoint() * U(site_mud, nu).adjoint() * U(site_mud, mu);
}

// Bottom left quadrant, i.e., P_{-mu, -nu}
[[nodiscard]]
Matrix_SU3 PlaquetteIII(const GaugeField& U, const site_coord& current_site, const int mu, const int nu) noexcept
{
    // TODO: We could reorder the expression below and replace one U.Move<-1> by a U.Move<1>
    //       Also, we could replace the last U.Move<-1> with a U.Move<1> from a different site
    site_coord site_mud     {U.Move<-1>(current_site, mu)};
    site_coord site_mud_nud {U.Move<-1>(site_mud    , nu)};
    return U(site_mud, mu).adjoint() * U(site_mud_nud, nu).adjoint() * U(site_mud_nud, mu) * U(U.Move<-1>(current_site, nu), nu);
}

// Bottom right quadrant, i.e., P_{-nu, mu}
[[nodiscard]]
Matrix_SU3 PlaquetteIV(const GaugeField& U, const site_coord& current_site, const int mu, const int nu) noexcept
{
    site_coord site_nud     {U.Move<-1>(current_site, nu)};
    return U(site_nud, nu).adjoint() * U(site_nud, mu) * U(U.Move<1>(site_nud, mu), nu) * U(current_site, mu).adjoint();
}

#endif // LETTUCE_PLAQUETTE_HPP
