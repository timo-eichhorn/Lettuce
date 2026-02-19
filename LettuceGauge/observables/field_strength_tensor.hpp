#ifndef LETTUCE_FIELD_STRENGTH_TENSOR_HPP
#define LETTUCE_FIELD_STRENGTH_TENSOR_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../math/su3.hpp"
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

//+---------------------------------------------------------------------------------+
//| This file provides implementations of the plaquette-based and clover-based      |
//| field strength tensor, as well as functions calculating the action/energy       |
//| density based on these definitions.                                             |
//+---------------------------------------------------------------------------------+

namespace FieldStrengthTensor
{
    // Version where the entries of F are antihermitian only (not traceless)
    void Clover(const GaugeField& U, FullTensor& F) noexcept
    {
        #pragma omp parallel for collapse(2)
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        {
            site_coord current_site {t, x, y, z};
            Matrix_3x3 Clover;
            F(current_site, 0, 0).setZero();
            Clover                = CalculateCloverComponent<1>(U, current_site, 0, 1);
            F(current_site, 0, 1) = -i<floatT> / 4.0 * SU3::Projection::Antihermitian(Clover);
            F(current_site, 1, 0) = -F(t, x, y, z, 0, 1).adjoint();

            Clover                = CalculateCloverComponent<1>(U, current_site, 0, 2);
            F(current_site, 0, 2) = -i<floatT> / 4.0 * SU3::Projection::Antihermitian(Clover);
            F(current_site, 2, 0) = -F(t, x, y, z, 0, 2).adjoint();

            Clover                = CalculateCloverComponent<1>(U, current_site, 0, 3);
            F(current_site, 0, 3) = -i<floatT> / 4.0 * SU3::Projection::Antihermitian(Clover);
            F(current_site, 3, 0) = -F(t, x, y, z, 0, 3).adjoint();

            F(current_site, 1, 1).setZero();
            Clover                = CalculateCloverComponent<1>(U, current_site, 1, 2);
            F(current_site, 1, 2) = -i<floatT> / 4.0 * SU3::Projection::Antihermitian(Clover);
            F(current_site, 2, 1) = -F(t, x, y, z, 1, 2).adjoint();

            Clover                = CalculateCloverComponent<1>(U, current_site, 1, 3);
            F(current_site, 1, 3) = -i<floatT> / 4.0 * SU3::Projection::Antihermitian(Clover);
            F(current_site, 3, 1) = -F(t, x, y, z, 1, 3).adjoint();

            F(current_site, 2, 2).setZero();
            Clover                = CalculateCloverComponent<1>(U, current_site, 2, 3);
            F(current_site, 2, 3) = -i<floatT> / 4.0 * SU3::Projection::Antihermitian(Clover);
            F(current_site, 3, 2) = -F(t, x, y, z, 2, 3).adjoint();
            F(current_site, 3, 3).setZero();
        }
    }
    // Version where the entries of F are made antihermitian and traceless (i.e. algebra elements)
    void CloverTraceless(const GaugeField& U, FullTensor& F) noexcept
    {
        #pragma omp parallel for collapse(2)
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        {
            site_coord current_site {t, x, y, z};
            Matrix_3x3 Clover;
            F(current_site, 0, 0).setZero();
            Clover                = CalculateCloverComponent<1>(U, current_site, 0, 1);
            F(current_site, 0, 1) = -i<floatT> / 4.0 * SU3::Projection::Algebra(Clover);
            F(current_site, 1, 0) = -F(t, x, y, z, 0, 1).adjoint();

            Clover                = CalculateCloverComponent<1>(U, current_site, 0, 2);
            F(current_site, 0, 2) = -i<floatT> / 4.0 * SU3::Projection::Algebra(Clover);
            F(current_site, 2, 0) = -F(t, x, y, z, 0, 2).adjoint();

            Clover                = CalculateCloverComponent<1>(U, current_site, 0, 3);
            F(current_site, 0, 3) = -i<floatT> / 4.0 * SU3::Projection::Algebra(Clover);
            F(current_site, 3, 0) = -F(t, x, y, z, 0, 3).adjoint();

            F(current_site, 1, 1).setZero();
            Clover                = CalculateCloverComponent<1>(U, current_site, 1, 2);
            F(current_site, 1, 2) = -i<floatT> / 4.0 * SU3::Projection::Algebra(Clover);
            F(current_site, 2, 1) = -F(t, x, y, z, 1, 2).adjoint();

            Clover                = CalculateCloverComponent<1>(U, current_site, 1, 3);
            F(current_site, 1, 3) = -i<floatT> / 4.0 * SU3::Projection::Algebra(Clover);
            F(current_site, 3, 1) = -F(t, x, y, z, 1, 3).adjoint();

            F(current_site, 2, 2).setZero();
            Clover                = CalculateCloverComponent<1>(U, current_site, 2, 3);
            F(current_site, 2, 3) = -i<floatT> / 4.0 * SU3::Projection::Algebra(Clover);
            F(current_site, 3, 2) = -F(t, x, y, z, 2, 3).adjoint();
            F(current_site, 3, 3).setZero();
        }
    }
    void MakeComponentsTraceless(FullTensor& F) noexcept
    {
        #pragma omp parallel for collapse(2)
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        {
            site_coord current_site {t, x, y, z};
            F(current_site, 0, 1) = SU3::Projection::Traceless(F(current_site, 0, 1));
            F(current_site, 1, 0) = -F(t, x, y, z, 0, 1).adjoint();

            F(current_site, 0, 2) = SU3::Projection::Traceless(F(current_site, 0, 2));
            F(current_site, 2, 0) = -F(t, x, y, z, 0, 2).adjoint();

            F(current_site, 0, 3) = SU3::Projection::Traceless(F(current_site, 0, 3));
            F(current_site, 3, 0) = -F(t, x, y, z, 0, 3).adjoint();

            F(current_site, 1, 2) = SU3::Projection::Traceless(F(current_site, 1, 2));
            F(current_site, 2, 1) = -F(t, x, y, z, 1, 2).adjoint();

            F(current_site, 1, 3) = SU3::Projection::Traceless(F(current_site, 1, 3));
            F(current_site, 3, 1) = -F(t, x, y, z, 1, 3).adjoint();

            F(current_site, 2, 3) = SU3::Projection::Traceless(F(current_site, 2, 3));
            F(current_site, 3, 2) = -F(t, x, y, z, 2, 3).adjoint();
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
        return 2 * (6 * Ncolor - E / U.Volume());
    }

    [[nodiscard]]
    double PlaquetteTimeslice(const GaugeField& U, const int t) noexcept
    {
        double E {PlaquetteSumTimeslice(U, t)};
        return 2 * (6 * Ncolor / U.Length(0) - E / U.Volume());
    }

    [[nodiscard]]
    double Clover(const FullTensor& F) noexcept
    {
        double E {0.0};
        #pragma omp parallel for collapse(2) reduction(+: E)
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
        // TODO: Normalization for different Ncolor?
        return E / F.Volume();
        // This should match Stephan's definition
        // return 1.0 / (36.0 * F.Volume()) * E;
    }

    [[nodiscard]]
    double CloverTimeslice(const FullTensor& F, const int t) noexcept
    {
        double E {0.0};
        #pragma omp parallel for collapse(2) reduction(+: E)
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
        // TODO: Normalization for different Ncolor?
        return E / F.Volume();
    }
} // namespace EnergyDensity

#endif // LETTUCE_FIELD_STRENGTH_TENSOR_HPP
