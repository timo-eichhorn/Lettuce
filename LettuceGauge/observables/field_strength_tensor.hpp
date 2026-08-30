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
    void Clover(const GaugeField& U, AntisymmetricField& FieldStrengthTensor) noexcept
    {
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
                const Matrix_3x3 clover {CalculateCloverComponent<1>(U, current_site, mu, nu)};
                FieldStrengthTensor.IndependentComponent(current_site, mu, nu) = -i<floatT> / 4.0 * SU3::Projection::Antihermitian(clover);
            }
        }
    }
    // Version where the entries of F are made antihermitian and traceless (i.e. algebra elements)
    void CloverTraceless(const GaugeField& U, AntisymmetricField& FieldStrengthTensor) noexcept
    {
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
                const Matrix_3x3 clover {CalculateCloverComponent<1>(U, current_site, mu, nu)};
                FieldStrengthTensor.IndependentComponent(current_site, mu, nu) = -i<floatT> / 4.0 * SU3::Projection::Algebra(clover);
            }
        }
    }
    void MakeComponentsTraceless(AntisymmetricField& FieldStrengthTensor) noexcept
    {
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
                Matrix_3x3& component {FieldStrengthTensor.IndependentComponent(current_site, mu, nu)};
                component = SU3::Projection::Traceless(component);
            }
        }
    }
} // namespace FieldStrengthTensor

namespace EnergyDensity
{
    // TODO: Replace by/add function which computes the energy density from already calculated plaquettes?
    [[nodiscard]]
    double Plaquette(const GaugeField& U) noexcept
    {
        const double E {PlaquetteSum(U)};
        return 2 * (6 * Ncolor - E / U.Volume());
    }

    [[nodiscard]]
    double PlaquetteTimeslice(const GaugeField& U, const int t) noexcept
    {
        const double E {PlaquetteSumTimeslice(U, t)};
        return 2 * (6 * Ncolor / U.Length(0) - E / U.Volume());
    }

    [[nodiscard]]
    double Clover(const AntisymmetricField& FieldStrengthTensor) noexcept
    {
        double E {0.0};
        #pragma omp parallel for collapse(omp_collapse_depth) reduction(+: E)
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        {
            const site_coord current_site {t, x, y, z};
            // F_{nu,mu} = F_{mu,nu}^{\dagger}
            // F_{nu,mu} F_{nu,mu} = (F_{mu,nu} F_{mu,nu})^{\dagger}
            // Due to the real trace, we can simplify the sum to go over (mu < nu) instead of (mu, nu) and get a factor of two
            // ReTr(F_01^2 + F_02^2 + F_03^2 + F_12^2 + F_13^2 + F_23^2)
            for (int mu = 0;      mu < Ndim; ++mu)
            for (int nu = mu + 1; nu < Ndim; ++nu)
            {
                const Matrix_3x3& component {FieldStrengthTensor.IndependentComponent(current_site, mu, nu)};
                E += std::real((component * component).trace());
            }
        }
        // TODO: Normalization for different Ncolor?
        return E / FieldStrengthTensor.Volume();
        // This should match Stephan's definition
        // return 1.0 / (36.0 * FieldStrengthTensor.Volume()) * E;
    }

    [[nodiscard]]
    double CloverTimeslice(const AntisymmetricField& FieldStrengthTensor, const int t) noexcept
    {
        double E {0.0};
        #pragma omp parallel for collapse(omp_collapse_depth) reduction(+: E)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        {
            const site_coord current_site {t, x, y, z};
            // F_{nu,mu} = F_{mu,nu}^{\dagger}
            // F_{nu,mu} F_{nu,mu} = (F_{mu,nu} F_{mu,nu})^{\dagger}
            // Due to the real trace, we can simplify the sum to go over (mu < nu) instead of (mu, nu) and get a factor of two
            // ReTr(F_01^2 + F_02^2 + F_03^2 + F_12^2 + F_13^2 + F_23^2)
            for (int mu = 0;      mu < Ndim; ++mu)
            for (int nu = mu + 1; nu < Ndim; ++nu)
            {
                const Matrix_3x3& component {FieldStrengthTensor.IndependentComponent(current_site, mu, nu)};
                E += std::real((component * component).trace());
            }
        }
        // TODO: Normalization for different Ncolor?
        return E / FieldStrengthTensor.Volume();
    }
} // namespace EnergyDensity

#endif // LETTUCE_FIELD_STRENGTH_TENSOR_HPP
