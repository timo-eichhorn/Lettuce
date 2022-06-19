#ifndef LETTUCE_PLAQUETTE_HPP
#define LETTUCE_PLAQUETTE_HPP

// Non-standard library headers
#include "../defines.hpp"
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

//-----
// Calculates plaquette at given coordinates
// TODO: Rewrite this? Is this even correct? Why does it return a zero matrix for nu = 0?

[[nodiscard]]
// Matrix_SU3 Plaquette(const GaugeField& Gluon, const int t, const int x, const int y, const int z, const int mu, const int nu) noexcept
Matrix_SU3 Plaquette(const GaugeField& Gluon, const site_coord& current_site, const int mu, const int nu) noexcept
{
    return Gluon(current_site, mu) * Gluon(Move<1>(current_site, mu), nu) * Gluon(Move<1>(current_site, nu), mu).adjoint() * Gluon(current_site, nu).adjoint();
}

// Top right quadrant, i.e., P_{mu, nu}
[[nodiscard]]
Matrix_SU3 PlaquetteI(const GaugeField& Gluon, const site_coord& current_site, const int mu, const int nu) noexcept
{
    return Gluon(current_site, mu) * Gluon(Move<1>(current_site, mu), nu) * Gluon(Move<1>(current_site, nu), mu).adjoint() * Gluon(current_site, nu).adjoint();
}

// Top left quadrant, i.e., P_{nu, -mu}
[[nodiscard]]
Matrix_SU3 PlaquetteII(const GaugeField& Gluon, const site_coord& current_site, const int mu, const int nu) noexcept
{
    site_coord site_mud     {Move<-1>(current_site, mu)};
    return Gluon(current_site, nu) * Gluon(Move<1>(site_mud, nu), mu).adjoint() * Gluon(site_mud, nu).adjoint() * Gluon(site_mud, mu);
}

// Bottom left quadrant, i.e., P_{-mu, -nu}
[[nodiscard]]
Matrix_SU3 PlaquetteIII(const GaugeField& Gluon, const site_coord& current_site, const int mu, const int nu) noexcept
{
    site_coord site_mud     {Move<-1>(current_site, mu)};
    site_coord site_mud_nud {Move<-1>(site_mud    , nu)};
    return Gluon(site_mud, mu).adjoint() * Gluon(site_mud_nud, nu).adjoint() * Gluon(site_mud_nud, mu) * Gluon(Move<-1>(current_site, nu), nu);
}

// Bottom right quadrant, i.e., P_{-nu, mu}
[[nodiscard]]
Matrix_SU3 PlaquetteIV(const GaugeField& Gluon, const site_coord& current_site, const int mu, const int nu) noexcept
{
    site_coord site_nud     {Move<-1>(current_site, nu)};
    return Gluon(site_nud, nu).adjoint() * Gluon(site_nud, mu) * Gluon(Move<1>(site_nud, mu), nu) * Gluon(current_site, mu).adjoint();
}

#endif // LETTUCE_PLAQUETTE_HPP
