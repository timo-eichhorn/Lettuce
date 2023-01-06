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

            F(current_site, 0, 0).setZero();
            Clover              = CalculateCloverComponent(U, current_site, 0, 1);
            // F(current_site, 0, 1) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
            // F(current_site, 0, 1) -= 1.0 / 3.0 * F(current_site, 0, 1).trace() * Matrix_3x3::Identity();
            F(current_site, 0, 1) = -i<floatT> / 4.0 * SU3::Projection::Algebra(Clover);
            F(current_site, 1, 0) = -F(t, x, y, z, 0, 1).adjoint();

            Clover              = CalculateCloverComponent(U, current_site, 0, 2);
            // F(current_site, 0, 2) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
            // F(current_site, 0, 2) -= 1.0 / 3.0 * F(current_site, 0, 2).trace() * Matrix_3x3::Identity();
            F(current_site, 0, 2) = -i<floatT> / 4.0 * SU3::Projection::Algebra(Clover);
            F(current_site, 2, 0) = -F(t, x, y, z, 0, 2).adjoint();

            Clover              = CalculateCloverComponent(U, current_site, 0, 3);
            // F(current_site, 0, 3) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
            // F(current_site, 0, 3) -= 1.0 / 3.0 * F(current_site, 0, 3).trace() * Matrix_3x3::Identity();
            F(current_site, 0, 3) = -i<floatT> / 4.0 * SU3::Projection::Algebra(Clover);
            F(current_site, 3, 0) = -F(t, x, y, z, 0, 3).adjoint();

            F(current_site, 1, 1).setZero();
            Clover              = CalculateCloverComponent(U, current_site, 1, 2);
            // F(current_site, 1, 2) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
            // F(current_site, 1, 2) -= 1.0 / 3.0 * F(current_site, 1, 2).trace() * Matrix_3x3::Identity();
            F(current_site, 1, 2) = -i<floatT> / 4.0 * SU3::Projection::Algebra(Clover);
            F(current_site, 2, 1) = -F(t, x, y, z, 1, 2).adjoint();

            Clover              = CalculateCloverComponent(U, current_site, 1, 3);
            // F(current_site, 1, 3) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
            // F(current_site, 1, 3) -= 1.0 / 3.0 * F(current_site, 1, 3).trace() * Matrix_3x3::Identity();
            F(current_site, 1, 3) = -i<floatT> / 4.0 * SU3::Projection::Algebra(Clover);
            F(current_site, 3, 1) = -F(t, x, y, z, 1, 3).adjoint();

            F(current_site, 2, 2).setZero();
            Clover              = CalculateCloverComponent(U, current_site, 2, 3);
            // F(current_site, 2, 3) = -i<floatT> / 8.0 * (Clover - Clover.adjoint());
            // F(current_site, 2, 3) -= 1.0 / 3.0 * F(current_site, 2, 3).trace() * Matrix_3x3::Identity();
            F(current_site, 2, 3) = -i<floatT> / 4.0 * SU3::Projection::Algebra(Clover);
            F(current_site, 3, 2) = -F(t, x, y, z, 2, 3).adjoint();
            F(current_site, 3, 3).setZero();
        }
    }
} // namespace FieldStrengthTensor

namespace EnergyDensity
{
    // TODO: Replace by/add function which computes the energy density from already calculated plaquettes?
    [[nodiscard]]
    double Plaquette(const GaugeField& U) noexcept
    {
        double E {PlaquetteSum(U)};
        return 36.0 - 2.0 * E / U.Volume();
    }

    [[nodiscard]]
    double Clover(const FullTensor& F) noexcept
    {
        double E {0.0};
        #pragma omp parallel for reduction(+: E)
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
        }
        // This should match Stephan's definition
        // return 1.0 / (36.0 * F.Volume()) * E;
        return E / F.Volume();
    }
} // namespace EnergyDensity

#endif // LETTUCE_FIELD_STRENGTH_TENSOR_HPP
