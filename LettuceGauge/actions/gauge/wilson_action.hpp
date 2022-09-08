#ifndef WILSON_ACTION_HPP
#define WILSON_ACTION_HPP

// Non-standard library headers
#include "../../defines.hpp"
#include "../../observables/plaquette.hpp"
#include <Eigen/Dense>
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <complex>
//----------------------------------------
// Standard C headers
// ...

namespace WilsonAction
{
    //-----
    // Returns unnormalized Wilson gauge action

    [[nodiscard]]
    double Action(const GaugeField& Gluon) noexcept
    {
        double S {0.0};

        #pragma omp parallel for reduction(+:S)
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int nu = 1; nu < 4; ++nu)
        {
            for (int mu = 0; mu < nu; ++mu)
            {
                S += std::real(Plaquette(Gluon, {t, x, y, z}, mu, nu).trace());
            }
        }
        return beta * (6.0 * Nt * Nx * Ny * Nz - 1.0/3.0 * S);
    }

    //-----
    // Returns normalized Wilson gauge action/Wilson gauge action per site

    [[nodiscard]]
    double ActionNormalized(const GaugeField& Gluon) noexcept
    {
        double S {0.0};
        // Matrix_SU3_double pl;

        #pragma omp parallel for reduction(+:S)
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int nu = 1; nu < 4; ++nu)
        {
            for (int mu = 0; mu < nu; ++mu)
            {
                // pl = Plaquette(Gluon, t, x, y, z, mu, nu);
                // S += 1.0/3.0 * std::real((Matrix_SU3::Identity() - pl).trace());
                // S += pl;
                // S += std::real(pl.trace());
                S += std::real(Plaquette(Gluon, {t, x, y, z}, mu, nu).trace());
                // pl += (Plaquette(Gluon, t, x, y, z, mu, nu)).cast<std::complex<double>>();
            }
        }
        // S = S/(6 * Nt * Nx * Ny * Nz);
        // return 1.0 - std::real(pl.trace())/18.0 * full_norm;
        return 1.0 - S/18.0 * full_norm;
    }

    //-----
    // Calculates staple at given coordinates
    // Compared to the definition used by Gattringer & Lang, this version is the adjoint

    [[nodiscard]]
    // Matrix_3x3 Staple(const GaugeField& Gluon, const int t, const int x, const int y, const int z, const int mu) noexcept
    Matrix_3x3 Staple(const GaugeField& Gluon, const site_coord& current_site, const int mu) noexcept
    {
        Matrix_3x3 st;

        // int tp = (t + 1)%Nt;
        // int tm = (t - 1 + Nt)%Nt;

        // int xp = (x + 1)%Nx;
        // int xm = (x - 1 + Nx)%Nx;

        // int yp = (y + 1)%Ny;
        // int ym = (y - 1 + Ny)%Ny;

        // int zp = (z + 1)%Nz;
        // int zm = (z - 1 + Nz)%Nz;

        auto [t, x, y, z] = current_site;

        switch(mu)
        {
            case 0:
            {
                int tp {(t + 1)%Nt};
                int xp {(x + 1)%Nx};
                int xm {(x - 1 + Nx)%Nx};
                int yp {(y + 1)%Ny};
                int ym {(y - 1 + Ny)%Ny};
                int zp {(z + 1)%Nz};
                int zm {(z - 1 + Nz)%Nz};
                st.noalias() = Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 0}) * Gluon({tp, x, y, z, 1}).adjoint() + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, z, 0}) * Gluon({tp, xm, y, z, 1})
                             + Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 0}) * Gluon({tp, x, y, z, 2}).adjoint() + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 0}) * Gluon({tp, x, ym, z, 2})
                             + Gluon({t, x, y, z, 3}) * Gluon({t, x, y, zp, 0}) * Gluon({tp, x, y, z, 3}).adjoint() + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 0}) * Gluon({tp, x, y, zm, 3});
                // st.noalias() = Gluon[t][x][y][z][1] * Gluon[t][xp][y][z][0] * Gluon[tp][x][y][z][1].adjoint() + Gluon[t][xm][y][z][1].adjoint() * Gluon[t][xm][y][z][0] * Gluon[tp][xm][y][z][1];
                // st.noalias() += Gluon[t][x][y][z][2] * Gluon[t][x][yp][z][0] * Gluon[tp][x][y][z][2].adjoint() + Gluon[t][x][ym][z][2].adjoint() * Gluon[t][x][ym][z][0] * Gluon[tp][x][ym][z][2];
                // st.noalias() += Gluon[t][x][y][z][3] * Gluon[t][x][y][zp][0] * Gluon[tp][x][y][z][3].adjoint() + Gluon[t][x][y][zm][3].adjoint() * Gluon[t][x][y][zm][0] * Gluon[tp][x][y][zm][3];
            }
            break;

            case 1:
            {
                int tp {(t + 1)%Nt};
                int tm {(t - 1 + Nt)%Nt};
                int xp {(x + 1)%Nx};
                int yp {(y + 1)%Ny};
                int ym {(y - 1 + Ny)%Ny};
                int zp {(z + 1)%Nz};
                int zm {(z - 1 + Nz)%Nz};
                st.noalias() = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 1}) * Gluon({t, xp, y, z, 0}).adjoint() + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 1}) * Gluon({tm, xp, y, z, 0})
                             + Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 1}) * Gluon({t, xp, y, z, 2}).adjoint() + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 1}) * Gluon({t, xp, ym, z, 2})
                             + Gluon({t, x, y, z, 3}) * Gluon({t, x, y, zp, 1}) * Gluon({t, xp, y, z, 3}).adjoint() + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 1}) * Gluon({t, xp, y, zm, 3});
                // st.noalias() = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 1}) * Gluon({t, xp, y, z, 0}).adjoint() + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 1}) * Gluon({tm, xp, y, z, 0});
                // st.noalias() += Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 1}) * Gluon({t, xp, y, z, 2}).adjoint() + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 1}) * Gluon({t, xp, ym, z, 2});
                // st.noalias() += Gluon({t, x, y, z, 3}) * Gluon({t, x, y, zp, 1}) * Gluon({t, xp, y, z, 3}).adjoint() + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 1}) * Gluon({t, xp, y, zm, 3});
            }
            break;

            case 2:
            {
                int tp {(t + 1)%Nt};
                int tm {(t - 1 + Nt)%Nt};
                int xp {(x + 1)%Nx};
                int xm {(x - 1 + Nx)%Nx};
                int yp {(y + 1)%Ny};
                int zp {(z + 1)%Nz};
                int zm {(z - 1 + Nz)%Nz};
                st.noalias() = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 2}) * Gluon({t, x, yp, z, 0}).adjoint() + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 2}) * Gluon({tm, x, yp, z, 0})
                             + Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 2}) * Gluon({t, x, yp, z, 1}).adjoint() + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, z, 2}) * Gluon({t, xm, yp, z, 1})
                             + Gluon({t, x, y, z, 3}) * Gluon({t, x, y, zp, 2}) * Gluon({t, x, yp, z, 3}).adjoint() + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 2}) * Gluon({t, x, yp, zm, 3});
                // st.noalias() = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 2}) * Gluon({t, x, yp, z, 0}).adjoint() + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 2}) * Gluon({tm, x, yp, z, 0});
                // st.noalias() += Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 2}) * Gluon({t, x, yp, z, 1}).adjoint() + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, z, 2}) * Gluon({t, xm, yp, z, 1});
                // st.noalias() += Gluon({t, x, y, z, 3}) * Gluon({t, x, y, zp, 2}) * Gluon({t, x, yp, z, 3}).adjoint() + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 2}) * Gluon({t, x, yp, zm, 3});
            }
            break;

            case 3:
            {
                int tp {(t + 1)%Nt};
                int tm {(t - 1 + Nt)%Nt};
                int xp {(x + 1)%Nx};
                int xm {(x - 1 + Nx)%Nx};
                int yp {(y + 1)%Ny};
                int ym {(y - 1 + Ny)%Ny};
                int zp {(z + 1)%Nz};
                st.noalias() = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 3}) * Gluon({t, x, y, zp, 0}).adjoint() + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 3}) * Gluon({tm, x, y, zp, 0})
                             + Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 3}) * Gluon({t, x, y, zp, 1}).adjoint() + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, z, 3}) * Gluon({t, xm, y, zp, 1})
                             + Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 3}) * Gluon({t, x, y, zp, 2}).adjoint() + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 3}) * Gluon({t, x, ym, zp, 2});
                // st.noalias() = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 3}) * Gluon({t, x, y, zp, 0}).adjoint() + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 3}) * Gluon({tm, x, y, zp, 0});
                // st.noalias() += Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 3}) * Gluon({t, x, y, zp, 1}).adjoint() + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, z, 3}) * Gluon({t, xm, y, zp, 1});
                // st.noalias() += Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 3}) * Gluon({t, x, y, zp, 2}).adjoint() + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 3}) * Gluon({t, x, ym, zp, 2});
            }
            break;
        }
        return st;
    }

    [[nodiscard]]
    // Matrix_3x3 Staple(const GaugeField& Gluon, const int t, const int x, const int y, const int z, const int mu) noexcept
    Matrix_3x3 Staple(const GaugeField& Gluon, const link_coord& current_link) noexcept
    {
        Matrix_3x3 st;
        auto [t, x, y, z, mu] = current_link;

        switch(mu)
        {
            case 0:
            {
                int tp {(t + 1)%Nt};
                int xp {(x + 1)%Nx};
                int xm {(x - 1 + Nx)%Nx};
                int yp {(y + 1)%Ny};
                int ym {(y - 1 + Ny)%Ny};
                int zp {(z + 1)%Nz};
                int zm {(z - 1 + Nz)%Nz};
                st.noalias() = Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 0}) * Gluon({tp, x, y, z, 1}).adjoint() + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, z, 0}) * Gluon({tp, xm, y, z, 1})
                             + Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 0}) * Gluon({tp, x, y, z, 2}).adjoint() + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 0}) * Gluon({tp, x, ym, z, 2})
                             + Gluon({t, x, y, z, 3}) * Gluon({t, x, y, zp, 0}) * Gluon({tp, x, y, z, 3}).adjoint() + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 0}) * Gluon({tp, x, y, zm, 3});
            }
            break;

            case 1:
            {
                int tp {(t + 1)%Nt};
                int tm {(t - 1 + Nt)%Nt};
                int xp {(x + 1)%Nx};
                int yp {(y + 1)%Ny};
                int ym {(y - 1 + Ny)%Ny};
                int zp {(z + 1)%Nz};
                int zm {(z - 1 + Nz)%Nz};
                st.noalias() = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 1}) * Gluon({t, xp, y, z, 0}).adjoint() + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 1}) * Gluon({tm, xp, y, z, 0})
                             + Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 1}) * Gluon({t, xp, y, z, 2}).adjoint() + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 1}) * Gluon({t, xp, ym, z, 2})
                             + Gluon({t, x, y, z, 3}) * Gluon({t, x, y, zp, 1}) * Gluon({t, xp, y, z, 3}).adjoint() + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 1}) * Gluon({t, xp, y, zm, 3});
            }
            break;

            case 2:
            {
                int tp {(t + 1)%Nt};
                int tm {(t - 1 + Nt)%Nt};
                int xp {(x + 1)%Nx};
                int xm {(x - 1 + Nx)%Nx};
                int yp {(y + 1)%Ny};
                int zp {(z + 1)%Nz};
                int zm {(z - 1 + Nz)%Nz};
                st.noalias() = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 2}) * Gluon({t, x, yp, z, 0}).adjoint() + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 2}) * Gluon({tm, x, yp, z, 0})
                             + Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 2}) * Gluon({t, x, yp, z, 1}).adjoint() + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, z, 2}) * Gluon({t, xm, yp, z, 1})
                             + Gluon({t, x, y, z, 3}) * Gluon({t, x, y, zp, 2}) * Gluon({t, x, yp, z, 3}).adjoint() + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 2}) * Gluon({t, x, yp, zm, 3});
            }
            break;

            case 3:
            {
                int tp {(t + 1)%Nt};
                int tm {(t - 1 + Nt)%Nt};
                int xp {(x + 1)%Nx};
                int xm {(x - 1 + Nx)%Nx};
                int yp {(y + 1)%Ny};
                int ym {(y - 1 + Ny)%Ny};
                int zp {(z + 1)%Nz};
                st.noalias() = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 3}) * Gluon({t, x, y, zp, 0}).adjoint() + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 3}) * Gluon({tm, x, y, zp, 0})
                             + Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 3}) * Gluon({t, x, y, zp, 1}).adjoint() + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, z, 3}) * Gluon({t, xm, y, zp, 1})
                             + Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 3}) * Gluon({t, x, y, zp, 2}).adjoint() + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 3}) * Gluon({t, x, ym, zp, 2});
            }
            break;
        }
        return st;
    }

    // [[nodiscard]]
    // Matrix_3x3 PartialStaple(const GaugeField& U, const site_coord& current_site, const int mu, const int nu) noexcept
    // {
    //     site_coord site_mup     {Move< 1>(current_site, mu)};
    //     site_coord site_nup     {Move< 1>(current_site, nu)};
    //     site_coord site_nud     {Move<-1>(current_site, nu)};
    //     site_coord site_mup_nud {Move<-1>(site_mup    , nu)};
    //     return U(current_site, nu)           * U(site_nup, mu) * U(site_mup    , nu).adjoint()
    //          + U(site_nud    , nu).adjoint() * U(site_nud, mu) * U(site_mup_nud, nu);
    // }

    // [[nodiscard]]
    // // Matrix_3x3 Staple(const GaugeField& U, const int t, const int x, const int y, const int z, const int mu) noexcept
    // Matrix_3x3 Staple(const GaugeField&U, const site_coord& current_site, const int mu) noexcept
    // {
    //     // site_coord current_site {t, x, y, z};
    //     // Matrix_3x3 st;
    //     // for (int nu_offset = 1; nu_offset < 4; ++nu_offset)
    //     // {
    //     //     int nu {(mu + nu_offset) % 4};
    //     //     site_coord site_mup     {Move< 1>(current_site, mu)};
    //     //     site_coord site_nup     {Move< 1>(current_site, nu)};
    //     //     site_coord site_nud     {Move<-1>(current_site, nu)};
    //     //     site_coord site_mup_nud {Move<-1>(site_mup    , nu)};
    //     //     st.noalias() += U(current_site, nu)           * U(site_nup, mu) * U(site_mup    , nu).adjoint()
    //     //                   + U(site_nud    , nu).adjoint() * U(site_nud, mu) * U(site_mup_nud, nu);
    //     // }
    //     int nu1 {(mu + 1)%4};
    //     int nu2 {(mu + 2)%4};
    //     int nu3 {(mu + 3)%4};
    //     site_coord site_mup      {Move< 1>(current_site, mu)};
    //     site_coord site_nup1     {Move< 1>(current_site, nu1)};
    //     site_coord site_nud1     {Move<-1>(current_site, nu1)};
    //     site_coord site_mup_nud1 {Move<-1>(site_mup    , nu1)};
    //     site_coord site_nup2     {Move< 1>(current_site, nu2)};
    //     site_coord site_nud2     {Move<-1>(current_site, nu2)};
    //     site_coord site_mup_nud2 {Move<-1>(site_mup    , nu2)};
    //     site_coord site_nup3     {Move< 1>(current_site, nu3)};
    //     site_coord site_nud3     {Move<-1>(current_site, nu3)};
    //     site_coord site_mup_nud3 {Move<-1>(site_mup    , nu3)};
    //     return U(current_site, nu1) * U(site_nup1, mu) * U(site_mup, nu1).adjoint() + U(site_nud1, nu1).adjoint() * U(site_nud1, mu) * U(site_mup_nud1, nu1)
    //          + U(current_site, nu2) * U(site_nup2, mu) * U(site_mup, nu2).adjoint() + U(site_nud2, nu2).adjoint() * U(site_nud2, mu) * U(site_mup_nud2, nu2)
    //          + U(current_site, nu3) * U(site_nup3, mu) * U(site_mup, nu3).adjoint() + U(site_nud3, nu3).adjoint() * U(site_nud3, mu) * U(site_mup_nud3, nu3);
    //     // return st;
    // }

    //-----

    // TODO: Does __restrict__ help in any way?
    // floatT SLocal(const Matrix_SU3&__restrict__ U, const Matrix_SU3&__restrict__ st)

    [[nodiscard]]
    double Local(const Matrix_SU3& U, const Matrix_3x3& st) noexcept
    {
        // return beta/3.0 * std::real((Matrix_SU3::Identity() - U * st.adjoint()).trace());
        return beta/static_cast<floatT>(3.0) * (static_cast<floatT>(3.0) - std::real((U * st.adjoint()).trace()));
    }

    // TODO: Does this help in any way?
    // [[nodiscard]]
    // floatT SLocalDiff(const Matrix_3x3& Udiff, const Matrix_3x3& st)
    // {
    //     return -beta/static_cast<floatT>(3.0) * std::real((Udiff * st.adjoint()).trace());
    // }
} // namespace WilsonAction

// namespace GaugeAction
// {
//     class Wilson
//     {
//         // Do not store the gauge field/a reference to the gauge field in the class, since we might want to use the same action for different fields (e.g. during smearing?)
//         // Instead, the field is always passed as an external reference
//         private:
//             double beta;
//         public:
//             // If we keep this constexpr, we can use it as template parameter in our coordinate move functions
//             static constexpr int stencil_radius {1};
//             //...
//             WilsonAction_(const double beta_in) noexcept :
//             beta(beta_in)
//             {}

//             void SetBeta(const double beta_in) noexcept
//             {
//                 beta = beta_in;
//             }

//             [[nodiscard]]
//             double GetBeta() noexcept
//             {
//                 return beta;
//             }

//             [[nodiscard]]
//             double ActionLocal() noexcept
//             {
//                 return;
//             }

//             [[nodiscard]]
//             double Action() noexcept
//             {
//                 return;
//             }

//             [[nodiscard]]
//             double ActionNormalized() noexcept
//             {
//                 return;
//             }

//             [[nodiscard]]
//             Matrix_3x3 Staple(const link_coord& link) noexcept
//             {
//                 return;
//             }
//     };
// }

#endif // WILSON_ACTION_HPP
