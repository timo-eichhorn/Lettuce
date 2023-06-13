#ifndef LETTUCE_POLYAKOV_LOOP_HPP
#define LETTUCE_POLYAKOV_LOOP_HPP

// Non-standard library headers
#include "../defines.hpp"
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

// With G++, OpenMP reduction on std::complex<T> doesn't work, so we have to define it ourselves
// #pragma omp declare \
// reduction(  \
//     + : \
//     std::complex<double> :  \
//     omp_out += omp_in ) \
// initializer( omp_priv = omp_orig )

[[nodiscard]]
std::complex<double> PolyakovLoop(const GaugeField& U) noexcept
{
    std::complex<floatT> P {0.0, 0.0};
    #pragma omp parallel for reduction(+: P)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        // std::complex<floatT> tmp {1.0, 0.0};
        Matrix_SU3 tmp {Matrix_SU3::Identity()};
        for (int t = 0; t < Nt; ++t)
        {
            tmp *= U({t, x, y, z, 0});
        }
        P += tmp.trace();
    }
    return P / U.SpatialVolume();
}

#endif // LETTUCE_POLYAKOV_LOOP_HPP
