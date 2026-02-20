#ifndef LETTUCE_PARITY_UPDATE_HPP
#define LETTUCE_PARITY_UPDATE_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../math/su3.hpp"
//-----
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

// TODO: Leaves action invariant and changes sign of clover charge (per timeslice even), but changes plaquette based charge?

// Perform an action preserving parity update that maps a configuration with topological charge Q to a configuration with charge -Q [hep-lat/0312035]
void ParityUpdate(GaugeField& U, GaugeField& U_copy) noexcept
{
    U_copy = U;

    #pragma omp parallel for collapse(omp_collapse_depth)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        const site_coord current_site {t, x, y, z};

        const int rx = Nx - x - 1;
        const int ry = Ny - y - 1;
        const int rz = Nz - z - 1;

        const int rx_m1 = (rx - 1 + Nx) % Nx;
        const int ry_m1 = (ry - 1 + Ny) % Ny;
        const int rz_m1 = (rz - 1 + Nz) % Nz;

        U(current_site, 0) = U_copy({t, rx,    ry,    rz,    0});
        U(current_site, 1) = U_copy({t, rx_m1, ry,    rz,    1}).adjoint();
        U(current_site, 2) = U_copy({t, rx,    ry_m1, rz,    2}).adjoint();
        U(current_site, 3) = U_copy({t, rx,    ry,    rz_m1, 3}).adjoint();
    }
}

#endif // LETTUCE_PARITY_UPDATE_HPP
