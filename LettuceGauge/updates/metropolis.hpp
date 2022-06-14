#ifndef LETTUCE_METROPOLIS_HPP
#define LETTUCE_METROPOLIS_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
// ...
//----------------------------------------
// Standard C headers
// ...

[[nodiscard]]
Matrix3x3 Staple(const GaugeField& U, const link_coord& coord)
{
    // int t = coord.t;
    // int x = coord.x;
    // int y = coord.y;
    // int z = coord.z;
    // TODO: Do in one line using structured bindings?
    // TODO: Does it make sense to declare as const?
    const auto [t, x, y, z, mu] = coord;   // This creates a copy
    // const auto& [t, x, y, z, mu] = coord;  // This creates references to the individual integers in coord
    switch(coord.mu)
    {
        case 0:
        {
            int tp = (t + 1)%Nt;
            int xp = (x + 1)%Nx;
            int xm = (x - 1 + Nx)%Nx;
            int yp = (y + 1)%Ny;
            int ym = (y - 1 + Ny)%Ny;
            int zp = (z + 1)%Nz;
            int zm = (z - 1 + Nz)%Nz;
            return U[t][x][y][z][1] * U[t][xp][y][z][0] * U[tp][x][y][z][1].adjoint() + U[t][xm][y][z][1].adjoint() * U[t][xm][y][z][0] * U[tp][xm][y][z][1]
                 + U[t][x][y][z][2] * U[t][x][yp][z][0] * U[tp][x][y][z][2].adjoint() + U[t][x][ym][z][2].adjoint() * U[t][x][ym][z][0] * U[tp][x][ym][z][2]
                 + U[t][x][y][z][3] * U[t][x][y][zp][0] * U[tp][x][y][z][3].adjoint() + U[t][x][y][zm][3].adjoint() * U[t][x][y][zm][0] * U[tp][x][y][zm][3];
        }
        break;

        case 1:
        {
            int tp = (t + 1)%Nt;
            int tm = (t - 1 + Nt)%Nt;
            int xp = (x + 1)%Nx;
            int yp = (y + 1)%Ny;
            int ym = (y - 1 + Ny)%Ny;
            int zp = (z + 1)%Nz;
            int zm = (z - 1 + Nz)%Nz;
            return U[t][x][y][z][0] * U[tp][x][y][z][1] * U[t][xp][y][z][0].adjoint() + U[tm][x][y][z][0].adjoint() * U[tm][x][y][z][1] * U[tm][xp][y][z][0]
                 + U[t][x][y][z][2] * U[t][x][yp][z][1] * U[t][xp][y][z][2].adjoint() + U[t][x][ym][z][2].adjoint() * U[t][x][ym][z][1] * U[t][xp][ym][z][2]
                 + U[t][x][y][z][3] * U[t][x][y][zp][1] * U[t][xp][y][z][3].adjoint() + U[t][x][y][zm][3].adjoint() * U[t][x][y][zm][1] * U[t][xp][y][zm][3];
        }
        break;

        case 2:
        {
            int tp = (t + 1)%Nt;
            int tm = (t - 1 + Nt)%Nt;
            int xp = (x + 1)%Nx;
            int xm = (x - 1 + Nx)%Nx;
            int yp = (y + 1)%Ny;
            int zp = (z + 1)%Nz;
            int zm = (z - 1 + Nz)%Nz;
            return U[t][x][y][z][0] * U[tp][x][y][z][2] * U[t][x][yp][z][0].adjoint() + U[tm][x][y][z][0].adjoint() * U[tm][x][y][z][2] * U[tm][x][yp][z][0]
                 + U[t][x][y][z][1] * U[t][xp][y][z][2] * U[t][x][yp][z][1].adjoint() + U[t][xm][y][z][1].adjoint() * U[t][xm][y][z][2] * U[t][xm][yp][z][1]
                 + U[t][x][y][z][3] * U[t][x][y][zp][2] * U[t][x][yp][z][3].adjoint() + U[t][x][y][zm][3].adjoint() * U[t][x][y][zm][2] * U[t][x][yp][zm][3];
        }
        break;

        case 3:
        {
            int tp = (t + 1)%Nt;
            int tm = (t - 1 + Nt)%Nt;
            int xp = (x + 1)%Nx;
            int xm = (x - 1 + Nx)%Nx;
            int yp = (y + 1)%Ny;
            int ym = (y - 1 + Ny)%Ny;
            int zp = (z + 1)%Nz;
            return U[t][x][y][z][0] * U[tp][x][y][z][3] * U[t][x][y][zp][0].adjoint() + U[tm][x][y][z][0].adjoint() * U[tm][x][y][z][3] * U[tm][x][y][zp][0]
                 + U[t][x][y][z][1] * U[t][xp][y][z][3] * U[t][x][y][zp][1].adjoint() + U[t][xm][y][z][1].adjoint() * U[t][xm][y][z][3] * U[t][xm][y][zp][1]
                 + U[t][x][y][z][2] * U[t][x][yp][z][3] * U[t][x][y][zp][2].adjoint() + U[t][x][ym][z][2].adjoint() * U[t][x][ym][z][3] * U[t][x][ym][zp][2];
        }
        break;
    }
}

void MetropolisSingle(GaugeField& U, const link_coord& coord)
{
    // Calculate staple
    // placeholder;
}

void MetropolisSweep()
{
    // placeholder;
}

#endif // LETTUCE_METROPOLIS_HPP
