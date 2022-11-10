#ifndef LETTUCE_FIELD_STRENGTH_TENSOR_HPP
#define LETTUCE_FIELD_STRENGTH_TENSOR_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "clover.hpp"
#include "plaquette.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <complex>
//----------------------------------------
// Standard C headers
// ...

 
namespace FieldStrengthTensor
{
    void Clover(const GaugeField& U, FullTensor& F) noexcept
    {
        #pragma omp parallel for
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        {
            site_coord current_site {t, x, y, z};
            Matrix_3x3 Clover;

            F(t, x, y, z, 0, 0).setZero();
            Clover              = CalculateCloverComponent(U, current_site, 0, 1);
            F(t, x, y, z, 0, 1) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
            F(t, x, y, z, 1, 0) = -F(t, x, y, z, 0, 1).adjoint();

            Clover              = CalculateCloverComponent(U, current_site, 0, 2);
            F(t, x, y, z, 0, 2) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
            F(t, x, y, z, 2, 0) = -F(t, x, y, z, 0, 2).adjoint();

            Clover              = CalculateCloverComponent(U, current_site, 0, 3);
            F(t, x, y, z, 0, 3) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
            F(t, x, y, z, 3, 0) = -F(t, x, y, z, 0, 3).adjoint();

            F(t, x, y, z, 1, 1).setZero();
            Clover              = CalculateCloverComponent(U, current_site, 1, 2);
            F(t, x, y, z, 1, 2) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
            F(t, x, y, z, 2, 1) = -F(t, x, y, z, 1, 2).adjoint();

            Clover              = CalculateCloverComponent(U, current_site, 1, 3);
            F(t, x, y, z, 1, 3) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
            F(t, x, y, z, 3, 1) = -F(t, x, y, z, 1, 3).adjoint();

            F(t, x, y, z, 2, 2).setZero();
            Clover              = CalculateCloverComponent(U, current_site, 2, 3);
            F(t, x, y, z, 2, 3) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
            F(t, x, y, z, 3, 2) = -F(t, x, y, z, 2, 3).adjoint();
            F(t, x, y, z, 3, 3).setZero();
        }
        // for (int mu = 0; mu < 4; ++mu)
        // for (int nu = 0; nu < 4; ++nu)
        // {
        //     site_coord current_site {t, x, y, z};
        //     Matrix_3x3 Clover      {CalculateCloverComponent(U, current_site, mu, nu)};
        //     F(t, x, y, z, mu, nu) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
        // }
    }
} // namespace FieldStrengthTensor

namespace EnergyDensity
{
    // TODO: Replace by/add function which computes the energy density from already calculates plaquettes?
    [[nodiscard]]
    double Plaquette(const GaugeField& U) noexcept
    {
        // double E {0.0};
        // #pragma omp parallel for reduction(+:E)
        // for (int t = 0; t < Nt; ++t)
        // for (int x = 0; x < Nx; ++x)
        // for (int y = 0; y < Ny; ++y)
        // for (int z = 0; z < Nz; ++z)
        // {
        //     site_coord current_site {t, x, y, z};
        //     // E += std::real((F(current_site, 0, 1) * F(current_site, 0, 1) + F(current_site, 0, 2) * F(current_site, 0, 2) + F(current_site, 0, 3) * F(current_site, 0, 3)
        //     //               + F(current_site, )).trace())
        // }
        double E {PlaquetteSum(U)};
        return 6.0 - 1.0 / (3.0 * U.Volume()) * E;
    }

    [[nodiscard]]
    double Clover(const FullTensor& F) noexcept
    {
        double E {0.0};
        #pragma omp parallel for reduction(+:E)
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        {
            site_coord current_site {t, x, y, z};
            // F_{nu,mu} = F_{mu,nu}^{\dagger}
            // F_{nu,mu} F_{nu,mu} = (F_{mu,nu} F_{mu,nu})^{\dagger}
            // Due to the real trace, we can simplify the sum to go over (mu < nu) instead of (mu, nu) and get a factor of two
            E += std::real((F(current_site, 0, 1) * F(current_site, 0, 1) + F(current_site, 0, 2) * F(current_site, 0, 2) + F(current_site, 0, 3) * F(current_site, 0, 3)
                          + F(current_site, 1, 2) * F(current_site, 1, 2) + F(current_site, 1, 3) * F(current_site, 1, 3) + F(current_site, 2, 3) * F(current_site, 2, 3)).trace());
            // E += std::real((F(current_site, 0, 1) * F(current_site, 1, 0) + F(current_site, 0, 2) * F(current_site, 2, 0) + F(current_site, 0, 3) * F(current_site, 3, 0)
            //               + F(current_site, 1, 2) * F(current_site, 2, 1) + F(current_site, 1, 3) * F(current_site, 3, 1) + F(current_site, 2, 3) * F(current_site, 3, 2)).trace());
        }
        // TODO: Factor 2.0 due to symmetry (see above)?
        //       Currently, this is missing an additional factor 0.5 compared to Stephan's definition
        //       Question is, what is the definition used for w_0?
        return 1.0 / (9.0 * 2.0 * F.Volume()) * E;
    }
} // namespace EnergyDensity

#endif // LETTUCE_FIELD_STRENGTH_TENSOR_HPP
