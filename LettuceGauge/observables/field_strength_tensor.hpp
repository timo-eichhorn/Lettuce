#ifndef LETTUCE_FIELD_STRENGTH_TENSOR_HPP
#define LETTUCE_FIELD_STRENGTH_TENSOR_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "clover.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <array>
#include <complex>
//----------------------------------------
// Standard C headers
// ...


namespace FieldStrengthTensor
{
    [[nodiscard]]
    void Clover(const GaugeField& U, FullTensor& F) noexcept
    {
        #pragma omp parallel for
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        for (int nu = 0; nu < 4; ++nu)
        {
            site_coord current_site {t, x, y, z};
            Matrix_3x3 C_mu_nu      {CalculateCloverComponent(U, current_site, mu, nu)};
            F(t, x, y, z, mu, nu) = -i<floatT> / 8.0 * (C_mu_nu - C_mu_nu.adjoint());
        }
    }
} // namespace FieldStrengthTensor

double EnergyDensity(const FullTensor& F) noexcept
{
    double E {0.0};
    #pragma omp parallel for reduction(+:E)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        // site_coord current_site {t, x, y, z};
        // E += std::real((F(current_site, 0, 1) * F(current_site, 0, 1) + F(current_site, 0, 2) * F(current_site, 0, 2) + F(current_site, 0, 3) * F(current_site, 0, 3)
        //               + F(current_site, )).trace())
    }
    return 1.0 / 4.0 * E;
}

#endif // LETTUCE_FIELD_STRENGTH_TENSOR_HPP
