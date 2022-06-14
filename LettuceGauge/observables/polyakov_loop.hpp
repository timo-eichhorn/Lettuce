#ifndef LETTUCE_POLYAKOV_LOOP_HPP
#define LETTUCE_POLYAKOV_LOOP_HPP

// Non-standard library headers
#include "../defines.hpp"
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

//-----
// Polyakov loops

[[nodiscard]]
std::complex<double> PolyakovLoop(const GaugeField& Gluon) noexcept
{
    std::complex<floatT> P {0.0, 0.0};
    #pragma omp parallel for reduction(+:P)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        // std::complex<floatT> tmp {1.0, 0.0};
        Matrix_SU3 tmp {Matrix_SU3::Identity()};
        for (int t = 0; t < Nt; ++t)
        {
            tmp *= Gluon({t, x, y, z, 0});
        }
        P += tmp.trace();
    }
    return P * spatial_norm;
}

#endif // LETTUCE_POLYAKOV_LOOP_HPP
