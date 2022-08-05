#ifndef LETTUCE_INSTANTON_HPP
#define LETTUCE_INSTANTON_HPP

// Non-standard library headers
#include "../defines.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
// ...
//----------------------------------------
// Standard C headers
#include <cmath>

// This sets the lattice equal to a BPST instanton
void CreateBPSTInstanton(GaugeField& Gluon, const site_coord& center, const int r) noexcept
{
    // TODO: Overload +/- operators on site_coord to calculate distances?
    // To avoid gauge singularities on the lattice, we actually do not place the instanton around center, but rather shift all coordinates by 0.5 into the positive direction
    // This way, the gauge singularity at the center of the instanton never actually coincides with a lattice point
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int mu = 0; mu < 4; ++mu)
    {
        site_coord current_site {t, x, y, z};
        double     distance2    {std::pow(current_site.t - (center.t + 0.5), 2) + std::pow(current_site.x - (center.x + 0.5), 2) + std::pow(current_site.y - (center.y + 0.5), 2) + std::pow(current_site.z - (center.z + 0.5), 2)};
        double     distance_mu  {current_site[mu] - (center[mu] + 0.5)};
        double     distance_mu2 {distance_mu * distance_mu};
        double     denom        {std::sqrt(distance2 - distance_mu2)};
        double     denom_rho    {std::sqrt(distance2 - distance_mu2 + r * r)}
        double     lambda       {-std::atan((distance_mu + 1.0) / denom) + std::atan(distance_mu / denom) + denom / denom_rho * (std::atan((distance_mu + 1.0) / denom_rho) - std::atan(distance_mu / denom_rho))};
        // For now only embed in 01 entries of SU(3) matrix
        Matrix_SU3 tmp {Matrix_SU3::Zero()};
        // for (int nu = 0; nu < 4; ++nu)
        // {
        //     if (mu != nu)
        //     {
        //         tmp +=
        //     }
        // }
        Matrix_SU3 sig1, sig2, sig3;
        sig1 << 0.0, 1.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, 0.0;
        sig2 << 0.0      , -i<floatT>, 0.0,
                i<floatT>,  0.0      , 0.0,
                0.0      ,  0.0      , 0.0;
        sig3 << 1.0,  0.0, 0.0,
                0.0, -1.0, 0.0,
                0.0,  0.0, 0.0;
        switch(mu)
        {
            case 0:
            {
                tmp += 
            }
            break;
            case 1:
            {
                //
            }
            break;
            case 2:
            {
                //
            }
            break;
            case 3:
            {
                //
            }
            break;
        }
        Gluon(current_site, mu) = std::cos(lambda) * Matrix_SU3::Identity() + i<floatT> * std::sin(lambda) / denom * tmp;
    }
}

#endif // LETTUCE_INSTANTON_HPP
