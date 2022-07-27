#ifndef LETTUCE_TOPOLOGICAL_CHARGE_HPP
#define LETTUCE_TOPOLOGICAL_CHARGE_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "plaquette.hpp"
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

//----------------------------------------
// Various (field-theoretic) definitions of the topological charge


//-----
// Calculate field-theoretic topological charge using field-strength tensor

[[nodiscard]]
double TopChargeGluonic(const GaugeField& Gluon) noexcept
{
    double Q {0.0};
    #pragma omp parallel for reduction(+:Q)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        Local_tensor Clov;
        Local_tensor F;
        int tm = (t - 1 + Nt)%Nt;
        int xm = (x - 1 + Nx)%Nx;
        int ym = (y - 1 + Ny)%Ny;
        int zm = (z - 1 + Nz)%Nz;
        int tp = (t + 1)%Nt;
        int xp = (x + 1)%Nx;
        int yp = (y + 1)%Ny;
        int zp = (z + 1)%Nz;
        // Calculate clover term
        // TODO: Rewrite using plaquette function?
        Clov[0][0].setZero();
        Clov[0][1] = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 1}) * Gluon({t, xp, y, z, 0}).adjoint() * Gluon({t, x, y, z, 1}).adjoint()
                   + Gluon({t, x, y, z, 1}) * Gluon({tm, xp, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 1}).adjoint() * Gluon({tm, x, y, z, 0})
                   + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, xm, y, z, 1}).adjoint() * Gluon({tm, xm, y, z, 0}) * Gluon({t, xm, y, z, 1})
                   + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, z, 0}) * Gluon({tp, xm, y, z, 1}) * Gluon({t, x, y, z, 0}).adjoint();
        Clov[1][0] = Clov[0][1].adjoint();

        Clov[0][2] = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 2}) * Gluon({t, x, yp, z, 0}).adjoint() * Gluon({t, x, y, z, 2}).adjoint()
                   + Gluon({t, x, y, z, 2}) * Gluon({tm, x, yp, z, 0}).adjoint() * Gluon({tm, x, y, z, 2}).adjoint() * Gluon({tm, x, y, z, 0})
                   + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, ym, z, 2}).adjoint() * Gluon({tm, x, ym, z, 0}) * Gluon({t, x, ym, z, 2})
                   + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 0}) * Gluon({tp, x, ym, z, 2}) * Gluon({t, x, y, z, 0}).adjoint();
        Clov[2][0] = Clov[0][2].adjoint();

        Clov[0][3] = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 3}) * Gluon({t, x, y, zp, 0}).adjoint() * Gluon({t, x, y, z, 3}).adjoint()
                   + Gluon({t, x, y, z, 3}) * Gluon({tm, x, y, zp, 0}).adjoint() * Gluon({tm, x, y, z, 3}).adjoint() * Gluon({tm, x, y, z, 0})
                   + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, zm, 3}).adjoint() * Gluon({tm, x, y, zm, 0}) * Gluon({t, x, y, zm, 3})
                   + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 0}) * Gluon({tp, x, y, zm, 3}) * Gluon({t, x, y, z, 0}).adjoint();
        Clov[3][0] = Clov[0][3].adjoint();

        Clov[1][1].setZero();
        Clov[1][2] = Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 2}) * Gluon({t, x, yp, z, 1}).adjoint() * Gluon({t, x, y, z, 2}).adjoint()
                   + Gluon({t, x, y, z, 2}) * Gluon({t, xm, yp, z, 1}).adjoint() * Gluon({t, xm, y, z, 2}).adjoint() * Gluon({t, xm, y, z, 1})
                   + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, ym, z, 2}).adjoint() * Gluon({t, xm, ym, z, 1}) * Gluon({t, x, ym, z, 2})
                   + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 1}) * Gluon({t, xp, ym, z, 2}) * Gluon({t, x, y, z, 1}).adjoint();
        Clov[2][1] = Clov[1][2].adjoint();

        Clov[1][3] = Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 3}) * Gluon({t, x, y, zp, 1}).adjoint() * Gluon({t, x, y, z, 3}).adjoint()
                   + Gluon({t, x, y, z, 3}) * Gluon({t, xm, y, zp, 1}).adjoint() * Gluon({t, xm, y, z, 3}).adjoint() * Gluon({t, xm, y, z, 1})
                   + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, zm, 3}).adjoint() * Gluon({t, xm, y, zm, 1}) * Gluon({t, x, y, zm, 3})
                   + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 1}) * Gluon({t, xp, y, zm, 3}) * Gluon({t, x, y, z, 1}).adjoint();
        Clov[3][1] = Clov[1][3].adjoint();

        Clov[2][2].setZero();
        Clov[2][3] = Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 3}) * Gluon({t, x, y, zp, 2}).adjoint() * Gluon({t, x, y, z, 3}).adjoint()
                   + Gluon({t, x, y, z, 3}) * Gluon({t, x, ym, zp, 2}).adjoint() * Gluon({t, x, ym, z, 3}).adjoint() * Gluon({t, x, ym, z, 2})
                   + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, zm, 3}).adjoint() * Gluon({t, x, ym, zm, 2}) * Gluon({t, x, y, zm, 3})
                   + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 2}) * Gluon({t, x, yp, zm, 3}) * Gluon({t, x, y, z, 2}).adjoint();
        Clov[3][2] = Clov[2][3].adjoint();
        Clov[3][3].setZero();
        // TODO: Use symmetry of F_mu,nu
        for (int mu = 0; mu < 4; ++mu)
        for (int nu = 0; nu < 4; ++nu)
        {
            F[mu][nu] = -i<floatT>/8.f * (Clov[mu][nu] - Clov[nu][nu]);
        }
        Q += std::real((F[0][1] * F[2][3] + F[0][2] * F[3][1] + F[0][3] * F[1][2]).trace());
    }
    return 1.0 / (4.0 * pi<double> * pi<double>) * Q;
}

[[nodiscard]]
double TopChargeGluonicSymm(const GaugeField& Gluon) noexcept
{
    double Q {0.0};
    #pragma omp parallel for reduction(+:Q)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        std::array<Matrix_3x3, 6> Clov;
        // TODO: SU3 or 3x3? Entries of F lie in the adjoint bundle, so probably not SU3?
        std::array<Matrix_3x3, 6> F;
        int tm = (t - 1 + Nt)%Nt;
        int xm = (x - 1 + Nx)%Nx;
        int ym = (y - 1 + Ny)%Ny;
        int zm = (z - 1 + Nz)%Nz;
        int tp = (t + 1)%Nt;
        int xp = (x + 1)%Nx;
        int yp = (y + 1)%Ny;
        int zp = (z + 1)%Nz;
        // site_coord current_site {t, x, y, z};
        // Calculate clover term using Q_mu,nu = Q_nu,mu^{dagger}
        // TODO: Rewrite using plaquette function?
        // Clov[0][0].setZero();
        Clov[0] = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 1}) * Gluon({t, xp, y, z, 0}).adjoint() * Gluon({t, x, y, z, 1}).adjoint()
                + Gluon({t, x, y, z, 1}) * Gluon({tm, xp, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 1}).adjoint() * Gluon({tm, x, y, z, 0})
                + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, xm, y, z, 1}).adjoint() * Gluon({tm, xm, y, z, 0}) * Gluon({t, xm, y, z, 1})
                + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, z, 0}) * Gluon({tp, xm, y, z, 1}) * Gluon({t, x, y, z, 0}).adjoint();
        // Clov[0] = PlaquetteI(Gluon, current_site, 0, 1) + PlaquetteII(Gluon, current_site, 0, 1) + PlaquetteIII(Gluon, current_site, 0, 1) + PlaquetteIV(Gluon, current_site, 0, 1);
        // Clov[1][0] = Clov[0][1].adjoint();

        Clov[1] = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 2}) * Gluon({t, x, yp, z, 0}).adjoint() * Gluon({t, x, y, z, 2}).adjoint()
                + Gluon({t, x, y, z, 2}) * Gluon({tm, x, yp, z, 0}).adjoint() * Gluon({tm, x, y, z, 2}).adjoint() * Gluon({tm, x, y, z, 0})
                + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, ym, z, 2}).adjoint() * Gluon({tm, x, ym, z, 0}) * Gluon({t, x, ym, z, 2})
                + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 0}) * Gluon({tp, x, ym, z, 2}) * Gluon({t, x, y, z, 0}).adjoint();
        // Clov[1] = PlaquetteI(Gluon, current_site, 0, 2) + PlaquetteII(Gluon, current_site, 0, 2) + PlaquetteIII(Gluon, current_site, 0, 2) + PlaquetteIV(Gluon, current_site, 0, 2);
        // Clov[2][0] = Clov[0][2].adjoint();

        Clov[2] = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 3}) * Gluon({t, x, y, zp, 0}).adjoint() * Gluon({t, x, y, z, 3}).adjoint()
                + Gluon({t, x, y, z, 3}) * Gluon({tm, x, y, zp, 0}).adjoint() * Gluon({tm, x, y, z, 3}).adjoint() * Gluon({tm, x, y, z, 0})
                + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, zm, 3}).adjoint() * Gluon({tm, x, y, zm, 0}) * Gluon({t, x, y, zm, 3})
                + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 0}) * Gluon({tp, x, y, zm, 3}) * Gluon({t, x, y, z, 0}).adjoint();
        // Clov[2] = PlaquetteI(Gluon, current_site, 0, 3) + PlaquetteII(Gluon, current_site, 0, 3) + PlaquetteIII(Gluon, current_site, 0, 3) + PlaquetteIV(Gluon, current_site, 0, 3);
        // Clov[3][0] = Clov[0][3].adjoint();

        // Clov[1]1.setZero();
        Clov[3] = Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 2}) * Gluon({t, x, yp, z, 1}).adjoint() * Gluon({t, x, y, z, 2}).adjoint()
                + Gluon({t, x, y, z, 2}) * Gluon({t, xm, yp, z, 1}).adjoint() * Gluon({t, xm, y, z, 2}).adjoint() * Gluon({t, xm, y, z, 1})
                + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, ym, z, 2}).adjoint() * Gluon({t, xm, ym, z, 1}) * Gluon({t, x, ym, z, 2})
                + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 1}) * Gluon({t, xp, ym, z, 2}) * Gluon({t, x, y, z, 1}).adjoint();
        // Clov[3] = PlaquetteI(Gluon, current_site, 1, 2) + PlaquetteII(Gluon, current_site, 1, 2) + PlaquetteIII(Gluon, current_site, 1, 2) + PlaquetteIV(Gluon, current_site, 1, 2);
        // Clov[2][1] = Clov[1][2].adjoint();

        Clov[4] = Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 3}) * Gluon({t, x, y, zp, 1}).adjoint() * Gluon({t, x, y, z, 3}).adjoint()
                + Gluon({t, x, y, z, 3}) * Gluon({t, xm, y, zp, 1}).adjoint() * Gluon({t, xm, y, z, 3}).adjoint() * Gluon({t, xm, y, z, 1})
                + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, zm, 3}).adjoint() * Gluon({t, xm, y, zm, 1}) * Gluon({t, x, y, zm, 3})
                + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 1}) * Gluon({t, xp, y, zm, 3}) * Gluon({t, x, y, z, 1}).adjoint();
        // Clov[4] = PlaquetteI(Gluon, current_site, 1, 3) + PlaquetteII(Gluon, current_site, 1, 3) + PlaquetteIII(Gluon, current_site, 1, 3) + PlaquetteIV(Gluon, current_site, 1, 3);
        // Clov[3][1] = Clov[1][3].adjoint();

        // Clov[2][2].setZero();
        Clov[5] = Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 3}) * Gluon({t, x, y, zp, 2}).adjoint() * Gluon({t, x, y, z, 3}).adjoint()
                + Gluon({t, x, y, z, 3}) * Gluon({t, x, ym, zp, 2}).adjoint() * Gluon({t, x, ym, z, 3}).adjoint() * Gluon({t, x, ym, z, 2})
                + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, zm, 3}).adjoint() * Gluon({t, x, ym, zm, 2}) * Gluon({t, x, y, zm, 3})
                + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 2}) * Gluon({t, x, yp, zm, 3}) * Gluon({t, x, y, z, 2}).adjoint();
        // Clov[5] = PlaquetteI(Gluon, current_site, 2, 3) + PlaquetteII(Gluon, current_site, 2, 3) + PlaquetteIII(Gluon, current_site, 2, 3) + PlaquetteIV(Gluon, current_site, 2, 3);
        // Clov[3][2] = Clov[2][3].adjoint();
        // Clov[3][3].setZero();
        // Version that uses the symmetry of F_mu,nu
        // for (int mu = 0; mu < 4; ++mu)
        // for (int nu = mu + 1; nu < 4; ++nu)
        // {
        //     F[mu][nu] = -i<floatT>/8.f * (Clov[mu][nu] - Clov[mu][nu].adjoint());
        // }
        //-----
        // // F[0][1]
        // F[0] = -i<floatT>/8.f * (Clov[0] - Clov[0].adjoint());
        // // F[0][2]
        // F[1] = -i<floatT>/8.f * (Clov[1] - Clov[1].adjoint());
        // // F[0][3]
        // F[2] = -i<floatT>/8.f * (Clov[2] - Clov[2].adjoint());
        // // F[1][2]
        // F[3] = -i<floatT>/8.f * (Clov[3] - Clov[3].adjoint());
        // // F[1][3]
        // F[4] = -i<floatT>/8.f * (Clov[4] - Clov[4].adjoint());
        // // F[2][3]
        // F[5] = -i<floatT>/8.f * (Clov[5] - Clov[5].adjoint());
        // Q += std::real((F[0] * F[5] - F[1] * F[4] + F[2] * F[3]).trace());
        //-----
        // F[0][1]
        F[0] = (Clov[0] - Clov[0].adjoint());
        // F[0][2]
        F[1] = (Clov[1] - Clov[1].adjoint());
        // F[0][3]
        F[2] = (Clov[2] - Clov[2].adjoint());
        // F[1][2]
        F[3] = (Clov[3] - Clov[3].adjoint());
        // F[1][3]
        F[4] = (Clov[4] - Clov[4].adjoint());
        // F[2][3]
        F[5] = (Clov[5] - Clov[5].adjoint());
        Q += std::real((F[0] * F[5] - F[1] * F[4] + F[2] * F[3]).trace());
    }
    return -1.0 / (64.0 * 4.0 * pi<double> * pi<double>) * Q;
}

[[nodiscard]]
double TopChargeGluonicSymm(const FullTensor& Clover) noexcept
{
    double Q {0.0};
    #pragma omp parallel for reduction(+:Q)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        std::array<Matrix_3x3, 6> F;
        //-----
        // F[0][1]
        F[0] = (Clover(t, x, y, z, 0, 1) - Clover(t, x, y, z, 1, 0));
        // F[0][2]
        F[1] = (Clover(t, x, y, z, 0, 2) - Clover(t, x, y, z, 2, 0));
        // F[0][3]
        F[2] = (Clover(t, x, y, z, 0, 3) - Clover(t, x, y, z, 3, 0));
        // F[1][2]
        F[3] = (Clover(t, x, y, z, 1, 2) - Clover(t, x, y, z, 2, 1));
        // F[1][3]
        F[4] = (Clover(t, x, y, z, 1, 3) - Clover(t, x, y, z, 3, 1));
        // F[2][3]
        F[5] = (Clover(t, x, y, z, 2, 3) - Clover(t, x, y, z, 3, 2));
        Q += std::real((F[0] * F[5] - F[1] * F[4] + F[2] * F[3]).trace());
    }
    return -1.0 / (64.0 * 4.0 * pi<double> * pi<double>) * Q;
}

[[nodiscard]]
double TopChargeGluonicUnimproved(const GaugeField& Gluon) noexcept
{
    double Q {0.0};
    #pragma omp parallel for reduction(+:Q)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        Local_tensor Clov;
        Local_tensor F;
        int tp = (t + 1)%Nt;
        int xp = (x + 1)%Nx;
        int yp = (y + 1)%Ny;
        int zp = (z + 1)%Nz;
        // Calculate clover term
        // TODO: Rewrite using plaquette function?
        Clov[0][0].setZero();
        Clov[0][1] = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 1}) * Gluon({t, xp, y, z, 0}).adjoint() * Gluon({t, x, y, z, 1}).adjoint();
        Clov[1][0] = Clov[0][1].adjoint();

        Clov[0][2] = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 2}) * Gluon({t, x, yp, z, 0}).adjoint() * Gluon({t, x, y, z, 2}).adjoint();
        Clov[2][0] = Clov[0][2].adjoint();

        Clov[0][3] = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 3}) * Gluon({t, x, y, zp, 0}).adjoint() * Gluon({t, x, y, z, 3}).adjoint();
        Clov[3][0] = Clov[0][3].adjoint();

        Clov[1][1].setZero();
        Clov[1][2] = Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 2}) * Gluon({t, x, yp, z, 1}).adjoint() * Gluon({t, x, y, z, 2}).adjoint();
        Clov[2][1] = Clov[1][2].adjoint();

        Clov[1][3] = Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 3}) * Gluon({t, x, y, zp, 1}).adjoint() * Gluon({t, x, y, z, 3}).adjoint();
        Clov[3][1] = Clov[1][3].adjoint();

        Clov[2][2].setZero();
        Clov[2][3] = Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 3}) * Gluon({t, x, y, zp, 2}).adjoint() * Gluon({t, x, y, z, 3}).adjoint();
        Clov[3][2] = Clov[2][3].adjoint();
        Clov[3][3].setZero();
        for (int mu = 0; mu < 4; ++mu)
        for (int nu = 0; nu < 4; ++nu)
        {
            F[mu][nu] = -i<floatT> * (Clov[mu][nu] - Clov[nu][mu]);
        }
        Q += std::real((F[0][1] * F[2][3] + F[0][2] * F[3][1] + F[0][3] * F[1][2]).trace());
    }
    // TODO: Normalization of 16.0 instead of 4.0 should yield correct results, but it is probably unnecessary to calcualte a clover term
    // like above. Instead, simply take the imaginary part of plaquettes and automatically get correct normalization?
    // For comparison of definitions see: https://arxiv.org/pdf/1708.00696.pdf
    return 1.0 / (16.0 * pi<double> * pi<double>) * Q;
}

//-----
// Calculate field-theoretic topological charge using field-strength tensor

// template<typename floatT, typename funcT>
// [[nodiscard]]
// double TopChargeGluonic(const Gl_Lattice& Gluon, funcT&& PlaqFunc)
// {
//     double Q {0.0};
//     #pragma omp parallel for reduction(+:Q)
//     for (int t = 0; t < Nt; ++t)
//     for (int x = 0; x < Nx; ++x)
//     for (int y = 0; y < Ny; ++y)
//     for (int z = 0; z < Nz; ++z)
//     {
//         Local_tensor Clov;
//         Local_tensor F;
//         int tm = (t - 1 + Nt)%Nt;
//         int xm = (x - 1 + Nx)%Nx;
//         int ym = (y - 1 + Ny)%Ny;
//         int zm = (z - 1 + Nz)%Nz;
//         int tp = (t + 1)%Nt;
//         int xp = (x + 1)%Nx;
//         int yp = (y + 1)%Ny;
//         int zp = (z + 1)%Nz;
//         // Calculate clover term
//         // TODO: Rewrite using plaquette function?
//         Clov[0][0].setZero();
//         Clov[0][1] = Gluon[t][x][y][z][0] * Gluon[tp][x][y][z][1] * Gluon[t][xp][y][z][0].adjoint() * Gluon[t][x][y][z][1].adjoint()
//                    + Gluon[t][x][y][z][1] * Gluon[tm][xp][y][z][0].adjoint() * Gluon[tm][x][y][z][1].adjoint() * Gluon[tm][x][y][z][0]
//                    + Gluon[tm][x][y][z][0].adjoint() * Gluon[tm][xm][y][z][1].adjoint() * Gluon[tm][xm][y][z][0] * Gluon[t][xm][y][z][1]
//                    + Gluon[t][xm][y][z][1].adjoint() * Gluon[t][xm][y][z][0] * Gluon[tp][xm][y][z][1] * Gluon[t][x][y][z][0].adjoint();
//         Clov[1][0] = Clov[0][1].adjoint();

//         Clov[0][2] = Gluon[t][x][y][z][0] * Gluon[tp][x][y][z][2] * Gluon[t][x][yp][z][0].adjoint() * Gluon[t][x][y][z][2].adjoint()
//                    + Gluon[t][x][y][z][2] * Gluon[tm][x][yp][z][0].adjoint() * Gluon[tm][x][y][z][2].adjoint() * Gluon[tm][x][y][z][0]
//                    + Gluon[tm][x][y][z][0].adjoint() * Gluon[tm][x][ym][z][2].adjoint() * Gluon[tm][x][ym][z][0] * Gluon[t][x][ym][z][2]
//                    + Gluon[t][x][ym][z][2].adjoint() * Gluon[t][x][ym][z][0] * Gluon[tp][x][ym][z][2] * Gluon[t][x][y][z][0].adjoint();
//         Clov[2][0] = Clov[0][2].adjoint();

//         Clov[0][3] = Gluon[t][x][y][z][0] * Gluon[tp][x][y][z][3] * Gluon[t][x][y][zp][0].adjoint() * Gluon[t][x][y][z][3].adjoint()
//                    + Gluon[t][x][y][z][3] * Gluon[tm][x][y][zp][0].adjoint() * Gluon[tm][x][y][z][3].adjoint() * Gluon[tm][x][y][z][0]
//                    + Gluon[tm][x][y][z][0].adjoint() * Gluon[tm][x][y][zm][3].adjoint() * Gluon[tm][x][y][zm][0] * Gluon[t][x][y][zm][3]
//                    + Gluon[t][x][y][zm][3].adjoint() * Gluon[t][x][y][zm][0] * Gluon[tp][x][y][zm][3] * Gluon[t][x][y][z][0].adjoint();
//         Clov[3][0] = Clov[0][3].adjoint();

//         Clov[1][1].setZero();
//         Clov[1][2] = Gluon[t][x][y][z][1] * Gluon[t][xp][y][z][2] * Gluon[t][x][yp][z][1].adjoint() * Gluon[t][x][y][z][2].adjoint()
//                    + Gluon[t][x][y][z][2] * Gluon[t][xm][yp][z][1].adjoint() * Gluon[t][xm][y][z][2].adjoint() * Gluon[t][xm][y][z][1]
//                    + Gluon[t][xm][y][z][1].adjoint() * Gluon[t][xm][ym][z][2].adjoint() * Gluon[t][xm][ym][z][1] * Gluon[t][x][ym][z][2]
//                    + Gluon[t][x][ym][z][2].adjoint() * Gluon[t][x][ym][z][1] * Gluon[t][xp][ym][z][2] * Gluon[t][x][y][z][1].adjoint();
//         Clov[2][1] = Clov[1][2].adjoint();

//         Clov[1][3] = Gluon[t][x][y][z][1] * Gluon[t][xp][y][z][3] * Gluon[t][x][y][zp][1].adjoint() * Gluon[t][x][y][z][3].adjoint()
//                    + Gluon[t][x][y][z][3] * Gluon[t][xm][y][zp][1].adjoint() * Gluon[t][xm][y][z][3].adjoint() * Gluon[t][xm][y][z][1]
//                    + Gluon[t][xm][y][z][1].adjoint() * Gluon[t][xm][y][zm][3].adjoint() * Gluon[t][xm][y][zm][1] * Gluon[t][x][y][zm][3]
//                    + Gluon[t][x][y][zm][3].adjoint() * Gluon[t][x][y][zm][1] * Gluon[t][xp][y][zm][3] * Gluon[t][x][y][z][1].adjoint();
//         Clov[3][1] = Clov[1][3].adjoint();

//         Clov[2][2].setZero();
//         Clov[2][3] = Gluon[t][x][y][z][2] * Gluon[t][x][yp][z][3] * Gluon[t][x][y][zp][2].adjoint() * Gluon[t][x][y][z][3].adjoint()
//                    + Gluon[t][x][y][z][3] * Gluon[t][x][ym][zp][2].adjoint() * Gluon[t][x][ym][z][3].adjoint() * Gluon[t][x][ym][z][2]
//                    + Gluon[t][x][ym][z][2].adjoint() * Gluon[t][x][ym][zm][3].adjoint() * Gluon[t][x][ym][zm][2] * Gluon[t][x][y][zm][3]
//                    + Gluon[t][x][y][zm][3].adjoint() * Gluon[t][x][y][zm][2] * Gluon[t][x][yp][zm][3] * Gluon[t][x][y][z][2].adjoint();
//         Clov[3][2] = Clov[2][3].adjoint();
//         Clov[3][3].setZero();
//         // TODO: Symmetry
//         for (int mu = 0; mu < 4; ++mu)
//         for (int nu = 0; nu < 4; ++nu)
//         {
//             F[mu][nu] = -i<float>/8.f * (Clov[mu][nu] - Clov[nu][mu]);
//         }
//         Q += std::real((F[0][1] * F[2][3] + F[0][2] * F[3][1] + F[0][3] * F[1][2]).trace());
//     }
//     return 1.0 / (4.0 * pi<double> * pi<double>) * Q;
// }

// TODO: Write this

// [[nodiscard]]
// double TopChargeGluonicImproved(const Gl_Lattice& Gluon)
// {
//     double Q {0.0};
//     #pragma omp parallel for reduction(+:Q)
//     for (int t = 0; t < Nt; ++t)
//     for (int x = 0; x < Nx; ++x)
//     for (int y = 0; y < Ny; ++y)
//     for (int z = 0; z < Nz; ++z)
//     {
//         std::array<Matrix_3x3, 6> Clov;
//         // TODO: SU3 or 3x3? Entries of F lie in the adjoint bundle, so probably not SU3?
//         std::array<Matrix_3x3, 6> F;
//         int tm = (t - 1 + Nt)%Nt;
//         int xm = (x - 1 + Nx)%Nx;
//         int ym = (y - 1 + Ny)%Ny;
//         int zm = (z - 1 + Nz)%Nz;
//         int tp = (t + 1)%Nt;
//         int xp = (x + 1)%Nx;
//         int yp = (y + 1)%Ny;
//         int zp = (z + 1)%Nz;
//         // Calculate clover term using Q_mu,nu = Q_nu,mu^{dagger}
//         // TODO: Rewrite using plaquette function?
//         // Clov[0][0].setZero();
//         Clov[0] = Gluon[t][x][y][z][0] * Gluon[tp][x][y][z][1] * Gluon[t][xp][y][z][0].adjoint() * Gluon[t][x][y][z][1].adjoint()
//                 + Gluon[t][x][y][z][1] * Gluon[tm][xp][y][z][0].adjoint() * Gluon[tm][x][y][z][1].adjoint() * Gluon[tm][x][y][z][0]
//                 + Gluon[tm][x][y][z][0].adjoint() * Gluon[tm][xm][y][z][1].adjoint() * Gluon[tm][xm][y][z][0] * Gluon[t][xm][y][z][1]
//                 + Gluon[t][xm][y][z][1].adjoint() * Gluon[t][xm][y][z][0] * Gluon[tp][xm][y][z][1] * Gluon[t][x][y][z][0].adjoint();
//         // Clov[1][0] = Clov[0][1].adjoint();

//         Clov[1] = Gluon[t][x][y][z][0] * Gluon[tp][x][y][z][2] * Gluon[t][x][yp][z][0].adjoint() * Gluon[t][x][y][z][2].adjoint()
//                 + Gluon[t][x][y][z][2] * Gluon[tm][x][yp][z][0].adjoint() * Gluon[tm][x][y][z][2].adjoint() * Gluon[tm][x][y][z][0]
//                 + Gluon[tm][x][y][z][0].adjoint() * Gluon[tm][x][ym][z][2].adjoint() * Gluon[tm][x][ym][z][0] * Gluon[t][x][ym][z][2]
//                 + Gluon[t][x][ym][z][2].adjoint() * Gluon[t][x][ym][z][0] * Gluon[tp][x][ym][z][2] * Gluon[t][x][y][z][0].adjoint();
//         // Clov[2][0] = Clov[0][2].adjoint();

//         Clov[2] = Gluon[t][x][y][z][0] * Gluon[tp][x][y][z][3] * Gluon[t][x][y][zp][0].adjoint() * Gluon[t][x][y][z][3].adjoint()
//                 + Gluon[t][x][y][z][3] * Gluon[tm][x][y][zp][0].adjoint() * Gluon[tm][x][y][z][3].adjoint() * Gluon[tm][x][y][z][0]
//                 + Gluon[tm][x][y][z][0].adjoint() * Gluon[tm][x][y][zm][3].adjoint() * Gluon[tm][x][y][zm][0] * Gluon[t][x][y][zm][3]
//                 + Gluon[t][x][y][zm][3].adjoint() * Gluon[t][x][y][zm][0] * Gluon[tp][x][y][zm][3] * Gluon[t][x][y][z][0].adjoint();
//         // Clov[3][0] = Clov[0][3].adjoint();

//         // Clov[1][1].setZero();
//         Clov[3] = Gluon[t][x][y][z][1] * Gluon[t][xp][y][z][2] * Gluon[t][x][yp][z][1].adjoint() * Gluon[t][x][y][z][2].adjoint()
//                 + Gluon[t][x][y][z][2] * Gluon[t][xm][yp][z][1].adjoint() * Gluon[t][xm][y][z][2].adjoint() * Gluon[t][xm][y][z][1]
//                 + Gluon[t][xm][y][z][1].adjoint() * Gluon[t][xm][ym][z][2].adjoint() * Gluon[t][xm][ym][z][1] * Gluon[t][x][ym][z][2]
//                 + Gluon[t][x][ym][z][2].adjoint() * Gluon[t][x][ym][z][1] * Gluon[t][xp][ym][z][2] * Gluon[t][x][y][z][1].adjoint();
//         // Clov[2][1] = Clov[1][2].adjoint();

//         Clov[4] = Gluon[t][x][y][z][1] * Gluon[t][xp][y][z][3] * Gluon[t][x][y][zp][1].adjoint() * Gluon[t][x][y][z][3].adjoint()
//                 + Gluon[t][x][y][z][3] * Gluon[t][xm][y][zp][1].adjoint() * Gluon[t][xm][y][z][3].adjoint() * Gluon[t][xm][y][z][1]
//                 + Gluon[t][xm][y][z][1].adjoint() * Gluon[t][xm][y][zm][3].adjoint() * Gluon[t][xm][y][zm][1] * Gluon[t][x][y][zm][3]
//                 + Gluon[t][x][y][zm][3].adjoint() * Gluon[t][x][y][zm][1] * Gluon[t][xp][y][zm][3] * Gluon[t][x][y][z][1].adjoint();
//         // Clov[3][1] = Clov[1][3].adjoint();

//         // Clov[2][2].setZero();
//         Clov[5] = Gluon[t][x][y][z][2] * Gluon[t][x][yp][z][3] * Gluon[t][x][y][zp][2].adjoint() * Gluon[t][x][y][z][3].adjoint()
//                 + Gluon[t][x][y][z][3] * Gluon[t][x][ym][zp][2].adjoint() * Gluon[t][x][ym][z][3].adjoint() * Gluon[t][x][ym][z][2]
//                 + Gluon[t][x][ym][z][2].adjoint() * Gluon[t][x][ym][zm][3].adjoint() * Gluon[t][x][ym][zm][2] * Gluon[t][x][y][zm][3]
//                 + Gluon[t][x][y][zm][3].adjoint() * Gluon[t][x][y][zm][2] * Gluon[t][x][yp][zm][3] * Gluon[t][x][y][z][2].adjoint();
//         // Clov[3][2] = Clov[2][3].adjoint();
//         // Clov[3][3].setZero();
//         //-----
//         // F[0][1]
//         F[0] = (Clov[0] - Clov[0].adjoint());
//         // F[0][2]
//         F[1] = (Clov[1] - Clov[1].adjoint());
//         // F[0][3]
//         F[2] = (Clov[2] - Clov[2].adjoint());
//         // F[1][2]
//         F[3] = (Clov[3] - Clov[3].adjoint());
//         // F[1][3]
//         F[4] = (Clov[4] - Clov[4].adjoint());
//         // F[2][3]
//         F[5] = (Clov[5] - Clov[5].adjoint());
//         Q += std::real((F[0] * F[5] - F[1] * F[4] + F[2] * F[3]).trace());
//     }
//     return -1.0 / (64.0 * 4.0 * pi<double> * pi<double>) * Q;
// }

// [[nodiscard]]
// double TopChargeGluonicOld(const Gl_Lattice& Gluon, Full_tensor& F)
// {
//     double Q {0.0};
//     #pragma omp parallel for reduction(+:Q)
//     for (int t = 0; t < Nt; ++t)
//     for (int x = 0; x < Nx; ++x)
//     for (int y = 0; y < Ny; ++y)
//     for (int z = 0; z < Nz; ++z)
//     {
//         Q += std::real((F[t][x][y][z][0][1] * F[t][x][y][z][2][3] + F[t][x][y][z][0][2] * F[t][x][y][z][3][1] + F[t][x][y][z][0][3] * F[t][x][y][z][1][2]).trace());
//     }
//     return 1.0 / (4.0 * pi<double> * pi<double>) * Q;
// }

#endif // LETTUCE_TOPOLOGICAL_CHARGE_HPP
