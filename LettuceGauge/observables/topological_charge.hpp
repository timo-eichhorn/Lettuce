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

//+---------------------------------------------------------------------------------+
//| This file provides functions to calculate the gluonic/field-theoretic           |
//| topological charge, either directly from the gauge field, or from an already    |
//| calculated clover-term or field strength tensor.                                |
//+---------------------------------------------------------------------------------+

[[deprecated("Use of this function is discouraged, preferably use TopChargeClover() instead!"), nodiscard]]
double TopChargeCloverSlow(const GaugeField& U) noexcept
{
    double Q {0.0};
    #pragma omp parallel for collapse(2) reduction(+: Q)
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
        Clov[0][1] = U({t, x, y, z, 0})            * U({tp, x, y, z, 1})            * U({t, xp, y, z, 0}).adjoint() * U({t, x, y, z, 1}).adjoint()
                   + U({t, x, y, z, 1})            * U({tm, xp, y, z, 0}).adjoint() * U({tm, x, y, z, 1}).adjoint() * U({tm, x, y, z, 0})
                   + U({tm, x, y, z, 0}).adjoint() * U({tm, xm, y, z, 1}).adjoint() * U({tm, xm, y, z, 0})          * U({t, xm, y, z, 1})
                   + U({t, xm, y, z, 1}).adjoint() * U({t, xm, y, z, 0})            * U({tp, xm, y, z, 1})          * U({t, x, y, z, 0}).adjoint();
        Clov[1][0] = Clov[0][1].adjoint();

        Clov[0][2] = U({t, x, y, z, 0})            * U({tp, x, y, z, 2})            * U({t, x, yp, z, 0}).adjoint() * U({t, x, y, z, 2}).adjoint()
                   + U({t, x, y, z, 2})            * U({tm, x, yp, z, 0}).adjoint() * U({tm, x, y, z, 2}).adjoint() * U({tm, x, y, z, 0})
                   + U({tm, x, y, z, 0}).adjoint() * U({tm, x, ym, z, 2}).adjoint() * U({tm, x, ym, z, 0})          * U({t, x, ym, z, 2})
                   + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, z, 0})            * U({tp, x, ym, z, 2})          * U({t, x, y, z, 0}).adjoint();
        Clov[2][0] = Clov[0][2].adjoint();

        Clov[0][3] = U({t, x, y, z, 0})            * U({tp, x, y, z, 3})            * U({t, x, y, zp, 0}).adjoint() * U({t, x, y, z, 3}).adjoint()
                   + U({t, x, y, z, 3})            * U({tm, x, y, zp, 0}).adjoint() * U({tm, x, y, z, 3}).adjoint() * U({tm, x, y, z, 0})
                   + U({tm, x, y, z, 0}).adjoint() * U({tm, x, y, zm, 3}).adjoint() * U({tm, x, y, zm, 0})          * U({t, x, y, zm, 3})
                   + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 0})            * U({tp, x, y, zm, 3})          * U({t, x, y, z, 0}).adjoint();
        Clov[3][0] = Clov[0][3].adjoint();

        Clov[1][1].setZero();
        Clov[1][2] = U({t, x, y, z, 1})            * U({t, xp, y, z, 2})            * U({t, x, yp, z, 1}).adjoint() * U({t, x, y, z, 2}).adjoint()
                   + U({t, x, y, z, 2})            * U({t, xm, yp, z, 1}).adjoint() * U({t, xm, y, z, 2}).adjoint() * U({t, xm, y, z, 1})
                   + U({t, xm, y, z, 1}).adjoint() * U({t, xm, ym, z, 2}).adjoint() * U({t, xm, ym, z, 1})          * U({t, x, ym, z, 2})
                   + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, z, 1})            * U({t, xp, ym, z, 2})          * U({t, x, y, z, 1}).adjoint();
        Clov[2][1] = Clov[1][2].adjoint();

        Clov[1][3] = U({t, x, y, z, 1})            * U({t, xp, y, z, 3})            * U({t, x, y, zp, 1}).adjoint() * U({t, x, y, z, 3}).adjoint()
                   + U({t, x, y, z, 3})            * U({t, xm, y, zp, 1}).adjoint() * U({t, xm, y, z, 3}).adjoint() * U({t, xm, y, z, 1})
                   + U({t, xm, y, z, 1}).adjoint() * U({t, xm, y, zm, 3}).adjoint() * U({t, xm, y, zm, 1})          * U({t, x, y, zm, 3})
                   + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 1})            * U({t, xp, y, zm, 3})          * U({t, x, y, z, 1}).adjoint();
        Clov[3][1] = Clov[1][3].adjoint();

        Clov[2][2].setZero();
        Clov[2][3] = U({t, x, y, z, 2})            * U({t, x, yp, z, 3})            * U({t, x, y, zp, 2}).adjoint() * U({t, x, y, z, 3}).adjoint()
                   + U({t, x, y, z, 3})            * U({t, x, ym, zp, 2}).adjoint() * U({t, x, ym, z, 3}).adjoint() * U({t, x, ym, z, 2})
                   + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, zm, 3}).adjoint() * U({t, x, ym, zm, 2})          * U({t, x, y, zm, 3})
                   + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 2})            * U({t, x, yp, zm, 3})          * U({t, x, y, z, 2}).adjoint();
        Clov[3][2] = Clov[2][3].adjoint();
        Clov[3][3].setZero();

        for (int mu = 0; mu < 4; ++mu)
        for (int nu = 0; nu < 4; ++nu)
        {
            F[mu][nu] = -i<floatT>/8.0 * (Clov[mu][nu] - Clov[nu][nu]);
        }
        Q += std::real((F[0][1] * F[2][3] + F[0][2] * F[3][1] + F[0][3] * F[1][2]).trace());
    }
    // Since we exploited the symmetries of the F_{mu,nu} term above, the normalization factor 1/32 turns into 1/4
    return 1.0 / (4.0 * pi<double> * pi<double>) * Q;
}

[[nodiscard]]
double TopChargeClover(const GaugeField& U) noexcept
{
    double Q {0.0};
    #pragma omp parallel for collapse(2) reduction(+: Q)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        std::array<Matrix_3x3, 6> Clov;
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
        // Calculate clover term using Q_{mu,nu} = Q_{nu,mu}^{dagger}
        // TODO: Rewrite using plaquette function?
        // Clov[0][1]
        Clov[0] = U({t, x, y, z, 0})            * U({tp, x, y, z, 1})            * U({t, xp, y, z, 0}).adjoint() * U({t, x, y, z, 1}).adjoint()
                + U({t, x, y, z, 1})            * U({tm, xp, y, z, 0}).adjoint() * U({tm, x, y, z, 1}).adjoint() * U({tm, x, y, z, 0})
                + U({tm, x, y, z, 0}).adjoint() * U({tm, xm, y, z, 1}).adjoint() * U({tm, xm, y, z, 0})          * U({t, xm, y, z, 1})
                + U({t, xm, y, z, 1}).adjoint() * U({t, xm, y, z, 0})            * U({tp, xm, y, z, 1})          * U({t, x, y, z, 0}).adjoint();
        // Clov[0] = PlaquetteI(U, current_site, 0, 1) + PlaquetteII(U, current_site, 0, 1) + PlaquetteIII(U, current_site, 0, 1) + PlaquetteIV(U, current_site, 0, 1);

        // Clov[0][2]
        Clov[1] = U({t, x, y, z, 0})            * U({tp, x, y, z, 2})            * U({t, x, yp, z, 0}).adjoint() * U({t, x, y, z, 2}).adjoint()
                + U({t, x, y, z, 2})            * U({tm, x, yp, z, 0}).adjoint() * U({tm, x, y, z, 2}).adjoint() * U({tm, x, y, z, 0})
                + U({tm, x, y, z, 0}).adjoint() * U({tm, x, ym, z, 2}).adjoint() * U({tm, x, ym, z, 0})          * U({t, x, ym, z, 2})
                + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, z, 0})            * U({tp, x, ym, z, 2})          * U({t, x, y, z, 0}).adjoint();
        // Clov[1] = PlaquetteI(U, current_site, 0, 2) + PlaquetteII(U, current_site, 0, 2) + PlaquetteIII(U, current_site, 0, 2) + PlaquetteIV(U, current_site, 0, 2);

        // Clov[0][3]
        Clov[2] = U({t, x, y, z, 0})            * U({tp, x, y, z, 3})            * U({t, x, y, zp, 0}).adjoint() * U({t, x, y, z, 3}).adjoint()
                + U({t, x, y, z, 3})            * U({tm, x, y, zp, 0}).adjoint() * U({tm, x, y, z, 3}).adjoint() * U({tm, x, y, z, 0})
                + U({tm, x, y, z, 0}).adjoint() * U({tm, x, y, zm, 3}).adjoint() * U({tm, x, y, zm, 0})          * U({t, x, y, zm, 3})
                + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 0})            * U({tp, x, y, zm, 3})          * U({t, x, y, z, 0}).adjoint();
        // Clov[2] = PlaquetteI(U, current_site, 0, 3) + PlaquetteII(U, current_site, 0, 3) + PlaquetteIII(U, current_site, 0, 3) + PlaquetteIV(U, current_site, 0, 3);

        // Clov[1][2]
        Clov[3] = U({t, x, y, z, 1})            * U({t, xp, y, z, 2})            * U({t, x, yp, z, 1}).adjoint() * U({t, x, y, z, 2}).adjoint()
                + U({t, x, y, z, 2})            * U({t, xm, yp, z, 1}).adjoint() * U({t, xm, y, z, 2}).adjoint() * U({t, xm, y, z, 1})
                + U({t, xm, y, z, 1}).adjoint() * U({t, xm, ym, z, 2}).adjoint() * U({t, xm, ym, z, 1})          * U({t, x, ym, z, 2})
                + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, z, 1})            * U({t, xp, ym, z, 2})          * U({t, x, y, z, 1}).adjoint();
        // Clov[3] = PlaquetteI(U, current_site, 1, 2) + PlaquetteII(U, current_site, 1, 2) + PlaquetteIII(U, current_site, 1, 2) + PlaquetteIV(U, current_site, 1, 2);

        // Clov[1][3]
        Clov[4] = U({t, x, y, z, 1})            * U({t, xp, y, z, 3})            * U({t, x, y, zp, 1}).adjoint() * U({t, x, y, z, 3}).adjoint()
                + U({t, x, y, z, 3})            * U({t, xm, y, zp, 1}).adjoint() * U({t, xm, y, z, 3}).adjoint() * U({t, xm, y, z, 1})
                + U({t, xm, y, z, 1}).adjoint() * U({t, xm, y, zm, 3}).adjoint() * U({t, xm, y, zm, 1})          * U({t, x, y, zm, 3})
                + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 1})            * U({t, xp, y, zm, 3})          * U({t, x, y, z, 1}).adjoint();
        // Clov[4] = PlaquetteI(U, current_site, 1, 3) + PlaquetteII(U, current_site, 1, 3) + PlaquetteIII(U, current_site, 1, 3) + PlaquetteIV(U, current_site, 1, 3);

        // Clov[2][3]
        Clov[5] = U({t, x, y, z, 2})            * U({t, x, yp, z, 3})            * U({t, x, y, zp, 2}).adjoint() * U({t, x, y, z, 3}).adjoint()
                + U({t, x, y, z, 3})            * U({t, x, ym, zp, 2}).adjoint() * U({t, x, ym, z, 3}).adjoint() * U({t, x, ym, z, 2})
                + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, zm, 3}).adjoint() * U({t, x, ym, zm, 2})          * U({t, x, y, zm, 3})
                + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 2})            * U({t, x, yp, zm, 3})          * U({t, x, y, z, 2}).adjoint();
        // Clov[5] = PlaquetteI(U, current_site, 2, 3) + PlaquetteII(U, current_site, 2, 3) + PlaquetteIII(U, current_site, 2, 3) + PlaquetteIV(U, current_site, 2, 3);

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
    // The factor -1/64 comes from the field strength tensor terms (factor -i/2 due to projection, and factor 1/4 due to 4 clover leaf terms)
    // Since we exploited the symmetries of the F_{mu,nu} term above, the normalization factor 1/32 turns into 1/4
    return -1.0 / (64.0 * 4.0 * pi<double> * pi<double>) * Q;
}

[[nodiscard]]
double TopChargeClover(const FullTensor& Clover) noexcept
{
    double Q {0.0};
    #pragma omp parallel for collapse(2) reduction(+: Q)
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
    // The factor -1/64 comes from the field strength tensor terms (factor -1/2 due to projection, and factor 1/4 due to 4 clover leaf terms)
    // Since we exploited the symmetries of the F_{mu,nu} term above, the normalization factor 1/32 turns into 1/4
    return -1.0 / (64.0 * 4.0 * pi<double> * pi<double>) * Q;
}

[[nodiscard]]
double TopChargeCloverTimeslice(const GaugeField& U, const int t) noexcept
{
    double Q {0.0};
    #pragma omp parallel for collapse(2) reduction(+: Q)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        std::array<Matrix_3x3, 6> Clov;
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
        // Calculate clover term using Q_{mu,nu} = Q_{nu,mu}^{dagger}
        // TODO: Rewrite using plaquette function?
        // Clov[0][1]
        Clov[0] = U({t, x, y, z, 0})            * U({tp, x, y, z, 1})            * U({t, xp, y, z, 0}).adjoint() * U({t, x, y, z, 1}).adjoint()
                + U({t, x, y, z, 1})            * U({tm, xp, y, z, 0}).adjoint() * U({tm, x, y, z, 1}).adjoint() * U({tm, x, y, z, 0})
                + U({tm, x, y, z, 0}).adjoint() * U({tm, xm, y, z, 1}).adjoint() * U({tm, xm, y, z, 0})          * U({t, xm, y, z, 1})
                + U({t, xm, y, z, 1}).adjoint() * U({t, xm, y, z, 0})            * U({tp, xm, y, z, 1})          * U({t, x, y, z, 0}).adjoint();
        // Clov[0] = PlaquetteI(U, current_site, 0, 1) + PlaquetteII(U, current_site, 0, 1) + PlaquetteIII(U, current_site, 0, 1) + PlaquetteIV(U, current_site, 0, 1);

        // Clov[0][2]
        Clov[1] = U({t, x, y, z, 0})            * U({tp, x, y, z, 2})            * U({t, x, yp, z, 0}).adjoint() * U({t, x, y, z, 2}).adjoint()
                + U({t, x, y, z, 2})            * U({tm, x, yp, z, 0}).adjoint() * U({tm, x, y, z, 2}).adjoint() * U({tm, x, y, z, 0})
                + U({tm, x, y, z, 0}).adjoint() * U({tm, x, ym, z, 2}).adjoint() * U({tm, x, ym, z, 0})          * U({t, x, ym, z, 2})
                + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, z, 0})            * U({tp, x, ym, z, 2})          * U({t, x, y, z, 0}).adjoint();
        // Clov[1] = PlaquetteI(U, current_site, 0, 2) + PlaquetteII(U, current_site, 0, 2) + PlaquetteIII(U, current_site, 0, 2) + PlaquetteIV(U, current_site, 0, 2);

        // Clov[0][3]
        Clov[2] = U({t, x, y, z, 0})            * U({tp, x, y, z, 3})            * U({t, x, y, zp, 0}).adjoint() * U({t, x, y, z, 3}).adjoint()
                + U({t, x, y, z, 3})            * U({tm, x, y, zp, 0}).adjoint() * U({tm, x, y, z, 3}).adjoint() * U({tm, x, y, z, 0})
                + U({tm, x, y, z, 0}).adjoint() * U({tm, x, y, zm, 3}).adjoint() * U({tm, x, y, zm, 0})          * U({t, x, y, zm, 3})
                + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 0})            * U({tp, x, y, zm, 3})          * U({t, x, y, z, 0}).adjoint();
        // Clov[2] = PlaquetteI(U, current_site, 0, 3) + PlaquetteII(U, current_site, 0, 3) + PlaquetteIII(U, current_site, 0, 3) + PlaquetteIV(U, current_site, 0, 3);

        // Clov[1][2]
        Clov[3] = U({t, x, y, z, 1})            * U({t, xp, y, z, 2})            * U({t, x, yp, z, 1}).adjoint() * U({t, x, y, z, 2}).adjoint()
                + U({t, x, y, z, 2})            * U({t, xm, yp, z, 1}).adjoint() * U({t, xm, y, z, 2}).adjoint() * U({t, xm, y, z, 1})
                + U({t, xm, y, z, 1}).adjoint() * U({t, xm, ym, z, 2}).adjoint() * U({t, xm, ym, z, 1})          * U({t, x, ym, z, 2})
                + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, z, 1})            * U({t, xp, ym, z, 2})          * U({t, x, y, z, 1}).adjoint();
        // Clov[3] = PlaquetteI(U, current_site, 1, 2) + PlaquetteII(U, current_site, 1, 2) + PlaquetteIII(U, current_site, 1, 2) + PlaquetteIV(U, current_site, 1, 2);

        // Clov[1][3]
        Clov[4] = U({t, x, y, z, 1})            * U({t, xp, y, z, 3})            * U({t, x, y, zp, 1}).adjoint() * U({t, x, y, z, 3}).adjoint()
                + U({t, x, y, z, 3})            * U({t, xm, y, zp, 1}).adjoint() * U({t, xm, y, z, 3}).adjoint() * U({t, xm, y, z, 1})
                + U({t, xm, y, z, 1}).adjoint() * U({t, xm, y, zm, 3}).adjoint() * U({t, xm, y, zm, 1})          * U({t, x, y, zm, 3})
                + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 1})            * U({t, xp, y, zm, 3})          * U({t, x, y, z, 1}).adjoint();
        // Clov[4] = PlaquetteI(U, current_site, 1, 3) + PlaquetteII(U, current_site, 1, 3) + PlaquetteIII(U, current_site, 1, 3) + PlaquetteIV(U, current_site, 1, 3);

        // Clov[2][3]
        Clov[5] = U({t, x, y, z, 2})            * U({t, x, yp, z, 3})            * U({t, x, y, zp, 2}).adjoint() * U({t, x, y, z, 3}).adjoint()
                + U({t, x, y, z, 3})            * U({t, x, ym, zp, 2}).adjoint() * U({t, x, ym, z, 3}).adjoint() * U({t, x, ym, z, 2})
                + U({t, x, ym, z, 2}).adjoint() * U({t, x, ym, zm, 3}).adjoint() * U({t, x, ym, zm, 2})          * U({t, x, y, zm, 3})
                + U({t, x, y, zm, 3}).adjoint() * U({t, x, y, zm, 2})            * U({t, x, yp, zm, 3})          * U({t, x, y, z, 2}).adjoint();
        // Clov[5] = PlaquetteI(U, current_site, 2, 3) + PlaquetteII(U, current_site, 2, 3) + PlaquetteIII(U, current_site, 2, 3) + PlaquetteIV(U, current_site, 2, 3);

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
    // The factor -1/64 comes from the field strength tensor terms (factor -i/2 due to projection, and factor 1/4 due to 4 clover leaf terms)
    // Since we exploited the symmetries of the F_{mu,nu} term above, the normalization factor 1/32 turns into 1/4
    return -1.0 / (64.0 * 4.0 * pi<double> * pi<double>) * Q;
}

[[nodiscard]]
double TopChargeCloverTimeslice(const FullTensor& Clover, const int t) noexcept
{
    double Q {0.0};
    #pragma omp parallel for collapse(2) reduction(+: Q)
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
    // Correct prefactor when using the clover term obtained from FieldStrengthTensor::Clover()
    return 1.0 / (16.0 * pi<double> * pi<double>) * Q;
}

[[nodiscard]]
double TopChargePlaquette(const GaugeField& U) noexcept
{
    double Q {0.0};
    #pragma omp parallel for collapse(2) reduction(+: Q)
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
        Clov[0][1] = U({t, x, y, z, 0}) * U({tp, x, y, z, 1}) * U({t, xp, y, z, 0}).adjoint() * U({t, x, y, z, 1}).adjoint();
        Clov[1][0] = Clov[0][1].adjoint();

        Clov[0][2] = U({t, x, y, z, 0}) * U({tp, x, y, z, 2}) * U({t, x, yp, z, 0}).adjoint() * U({t, x, y, z, 2}).adjoint();
        Clov[2][0] = Clov[0][2].adjoint();

        Clov[0][3] = U({t, x, y, z, 0}) * U({tp, x, y, z, 3}) * U({t, x, y, zp, 0}).adjoint() * U({t, x, y, z, 3}).adjoint();
        Clov[3][0] = Clov[0][3].adjoint();

        Clov[1][1].setZero();
        Clov[1][2] = U({t, x, y, z, 1}) * U({t, xp, y, z, 2}) * U({t, x, yp, z, 1}).adjoint() * U({t, x, y, z, 2}).adjoint();
        Clov[2][1] = Clov[1][2].adjoint();

        Clov[1][3] = U({t, x, y, z, 1}) * U({t, xp, y, z, 3}) * U({t, x, y, zp, 1}).adjoint() * U({t, x, y, z, 3}).adjoint();
        Clov[3][1] = Clov[1][3].adjoint();

        Clov[2][2].setZero();
        Clov[2][3] = U({t, x, y, z, 2}) * U({t, x, yp, z, 3}) * U({t, x, y, zp, 2}).adjoint() * U({t, x, y, z, 3}).adjoint();
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

[[nodiscard]]
double TopChargePlaquette2x2(const GaugeField& U) noexcept
{
    double Q {0.0};
    #pragma omp parallel for collapse(2) reduction(+: Q)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        std::array<Matrix_3x3, 6> Clov;
        std::array<Matrix_3x3, 6> F;
        site_coord                current_site {t, x, y, z};
        //-----
        Clov[0] = CalculateCloverComponent<2>(U, current_site, 0, 1);
        Clov[1] = CalculateCloverComponent<2>(U, current_site, 0, 2);
        Clov[2] = CalculateCloverComponent<2>(U, current_site, 0, 3);
        Clov[3] = CalculateCloverComponent<2>(U, current_site, 1, 2);
        Clov[4] = CalculateCloverComponent<2>(U, current_site, 1, 3);
        Clov[5] = CalculateCloverComponent<2>(U, current_site, 2, 3);
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
    // TODO: What is the correct normalization? Additional normalization factor 2/(2^2 * ^2^2)?
    return -1.0 / (64.0 * 4.0 * 8.0 * pi<double> * pi<double>) * Q;
}

// TODO: Not sure if this is correct yet, but the results don't seem completely incorrect at least
//       Also, does it even make sense to use different coefficients than the LW coefficients here? All other choices do not cancel the O(a^2) contributions (at tree-level)
double TopChargeCloverImproved(const GaugeField& U, const double c_plaq = 1.0 + 8.0 * 1.0/12.0/* = 5/3 */, const double c_rect = -1.0/12.0) noexcept
{
    double Q {0.0};
    #pragma omp parallel for collapse(2) reduction(+: Q)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        std::array<Matrix_3x3, 6> F_clover;
        std::array<Matrix_3x3, 6> F_rectangle;
        site_coord                current_site {t, x, y, z};
        //-----
        // F[0][1]
        F_clover[0]    = SU3::Projection::Antihermitian(CalculateCloverComponent<1, 1>(U, current_site, 0, 1));
        F_rectangle[0] = SU3::Projection::Antihermitian(CalculateCloverComponent<2, 1>(U, current_site, 0, 1)
                       +                                CalculateCloverComponent<1, 2>(U, current_site, 0, 1));
        // F[0][2]
        F_clover[1]    = SU3::Projection::Antihermitian(CalculateCloverComponent<1, 1>(U, current_site, 0, 2));
        F_rectangle[1] = SU3::Projection::Antihermitian(CalculateCloverComponent<2, 1>(U, current_site, 0, 2)
                       +                                CalculateCloverComponent<1, 2>(U, current_site, 0, 2));
        // F[0][3]
        F_clover[2]    = SU3::Projection::Antihermitian(CalculateCloverComponent<1, 1>(U, current_site, 0, 3));
        F_rectangle[2] = SU3::Projection::Antihermitian(CalculateCloverComponent<2, 1>(U, current_site, 0, 3)
                       +                                CalculateCloverComponent<1, 2>(U, current_site, 0, 3));
        // F[1][2]
        F_clover[3]    = SU3::Projection::Antihermitian(CalculateCloverComponent<1, 1>(U, current_site, 1, 2));
        F_rectangle[3] = SU3::Projection::Antihermitian(CalculateCloverComponent<2, 1>(U, current_site, 1, 2)
                       +                                CalculateCloverComponent<1, 2>(U, current_site, 1, 2));
        // F[1][3]
        F_clover[4]    = SU3::Projection::Antihermitian(CalculateCloverComponent<1, 1>(U, current_site, 1, 3));
        F_rectangle[4] = SU3::Projection::Antihermitian(CalculateCloverComponent<2, 1>(U, current_site, 1, 3)
                       +                                CalculateCloverComponent<1, 2>(U, current_site, 1, 3));
        // F[2][3]
        F_clover[5]    = SU3::Projection::Antihermitian(CalculateCloverComponent<1, 1>(U, current_site, 2, 3));
        F_rectangle[5] = SU3::Projection::Antihermitian(CalculateCloverComponent<2, 1>(U, current_site, 2, 3)
                       +                                CalculateCloverComponent<1, 2>(U, current_site, 2, 3));
        Q += c_plaq * std::real((F_clover[0] * F_clover[5] - F_clover[1] * F_clover[4] + F_clover[2] * F_clover[3]).trace())
           + 0.5 * c_rect * std::real((F_rectangle[0] * F_rectangle[5] - F_rectangle[1] * F_rectangle[4] + F_rectangle[2] * F_rectangle[3]).trace());
    }
    return -1.0 / (16.0 * 4.0 * pi<double> * pi<double>) * Q;
}

// TODO: WIP, still need to implement and check if all these functions make sense
namespace TopologicalCharge
{

    // [[nodiscard]]
    // double PlaquetteChargeFromGaugeField(const GaugeField& U) noexcept
    // {
    //     doubleQ {0.0};
    //     #pragma omp parallel for collapse(2) reduction(+: Q)
    //     for (int t = 0; t < Nt; ++t)
    //     for (int x = 0; x < Nx; ++x)
    //     for (int y = 0; y < Ny; ++y)
    //     for (int z = 0; z < Nz; ++z)
    //     {
    //         Local_tensor Clov;
    //         Local_tensor F;
    //         int tp = (t + 1)%Nt;
    //         int xp = (x + 1)%Nx;
    //         int yp = (y + 1)%Ny;
    //         int zp = (z + 1)%Nz;
    //     }
    // }

    // [[nodiscard]]
    // double PlaquetteChargeFromFTensor(const FullTensor& F) noexcept
    // {
    //     //
    // }

    // [[nodiscard]]
    // double CloverChargeFromGaugeField(const GaugeField& U) noexcept
    // {
    //     //
    // }

    // [[nodiscard]]
    // double CloverChargeFromClover(const FullTensor& Clover) noexcept
    // {
    //     //
    // }

    // TODO: Seems to work, but is inconsistent with the definition above
    //       Here, F is made antihermitian and traceless (due to the function calculating F), whereas above, F is not made traceless, which leads to slightly different results
    //       Using F without making it traceless gives the exact same results as the regular function above (as one would expect)
    [[nodiscard]]
    double CloverChargeFromFTensor(const FullTensor& F) noexcept
    {
        double Q {0.0};
        #pragma omp parallel for collapse(2) reduction(+: Q)
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        {
            site_coord current_site {t, x, y, z};
            Q += std::real((F(current_site, 0, 1) * F(current_site, 2, 3) + F(current_site, 0, 2) * F(current_site, 3, 1) + F(current_site, 0, 3) * F(current_site, 1, 2)).trace());
        }
        return 1.0 / (4.0 * pi<double> * pi<double>) * Q;
    }

    // [[nodiscard]]
    // double ImprovedCloverChargeFromGaugeField(const GaugeField& U) noexcept
    // {
    //     //
    // }

    // [[nodiscard]]
    // double ImprovedCloverChargeFromClover(const FullTensor& Clover_plaq, const FullTensor& Clover_rect) noexcept
    // {
    //     //
    // }

    // [[nodiscard]]
    // double ImprovedCloverChargeFromFTensor(const FullTensor& F) noexcept
    // {
    //     //
    // }
}

#endif // LETTUCE_TOPOLOGICAL_CHARGE_HPP
