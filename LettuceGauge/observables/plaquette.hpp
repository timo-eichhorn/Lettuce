#ifndef LETTUCE_PLAQUETTE_HPP
#define LETTUCE_PLAQUETTE_HPP

// Non-standard library headers
#include "../defines.hpp"
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

//-----
// Calculates plaquette at given coordinates
// TODO: Rewrite this? Is this even correct? Why does it return a zero matrix for nu = 0?

[[nodiscard]]
// Matrix_SU3 Plaquette(const GaugeField4D<Matrix_SU3>& Gluon, const int t, const int x, const int y, const int z, const int mu, const int nu) noexcept
Matrix_SU3 Plaquette(const GaugeField4D<Matrix_SU3>& Gluon, const site_coord& current_site, const int mu, const int nu) noexcept
{
    // site_coord current_site {t, x, y, z};
    return Gluon(current_site, mu) * Gluon(Move<1>(current_site, mu), nu) * Gluon(Move<1>(current_site, nu), mu).adjoint() * Gluon(current_site, nu).adjoint();
}


// template<typename floatT>
// [[nodiscard]]
// floatT Plaquette(const GField& U, const int t, const int x, const int y, const int z, const int mu, const int nu) noexcept
// {
//     SU3_matrix pl;

//     int tp = (t + 1)%Nt;
//     int xp = (x + 1)%Nx;
//     int yp = (y + 1)%Ny;
//     int zp = (z + 1)%Nz;

//     switch(nu)
//     {
//         case 0:
//         {
//             SU3_matrix pl << 0.0, 0.0, 0.0,
//                              0.0, 0.0, 0.0,
//                              0.0, 0.0, 0.0;
//         }
//         break;

//         case 1:
//         {
//             SU3_matrix pl = U[t][x][y][z][0] * U[tp][x][y][z][1] * U[t][xp][y][z][0].adjoint() * U[t][x][y][z][1].adjoint();
//         }
//         break;

//         case 2:
//         {
//             if (mu == 0)
//             {
//                 SU3_matrix pl = U[t][x][y][z][0] * U[tp][x][y][z][2] * U[t][x][yp][z][0].adjoint() * U[t][x][y][z][2].adjoint();
//             }
//             if (mu == 1)
//             {
//                 SU3_matrix pl = U[t][x][y][z][1] * U[t][xp][y][z][2] * U[t][x][yp][z][1].adjoint() * U[t][x][y][z][2].adjoint();
//             }
//         }
//         break;

//         case 3:
//         {
//             if(mu == 0)
//             {
//                 SU3_matrix pl = U[t][x][y][z][0] * U[tp][x][y][z][3] * U[t][x][y][zp][0].adjoint() * U[t][x][y][z][3].adjoint();
//             }
//             if(mu == 1)
//             {
//                 SU3_matrix pl = U[t][x][y][z][1] * U[t][xp][y][z][3] * U[t][x][y][zp][1].adjoint() * U[t][x][y][z][3].adjoint();
//             }
//             if(mu == 2)
//             {
//                 SU3_matrix pl = U[t][x][y][z][2] * U[t][x][yp][z][3] * U[t][x][y][zp][2].adjoint() * U[t][x][y][z][3].adjoint();
//             }
//         }
//         break;
//     }
//     return pl;
// }

#endif // LETTUCE_PLAQUETTE_HPP
