#ifndef LETTUCE_WILSON_LOOP_HPP
#define LETTUCE_WILSON_LOOP_HPP

// Non-standard library headers
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
// Square Wilson loops of length Nmu
// TODO: Does it make sense to precompute Gluonchain, or is it more efficient to locally calculate the terms?
//       For instance, we could left multiply with the adjoint/inverse and right multiply with a new link, which
//       might be computationally advantageous for larger chain lengths (only 2 multiplications instead of N)

template<int Nmu_start, int Nmu_end, bool reset>
[[nodiscard]]
double WilsonLoop(const GaugeField& Gluon, GaugeField& Gluonchain) noexcept
{
    double W {0.0};
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        if constexpr(reset)
        {
            for (int mu = 0; mu < 4; ++mu)
            {
                Gluonchain({t, x, y, z, mu}).setIdentity();
            }
        }
        for (int n = Nmu_start; n < Nmu_end; ++n)
        {
            Gluonchain({t, x, y, z, 0}) *= Gluon({(t + n)%Nt, x, y, z, 0});
            Gluonchain({t, x, y, z, 1}) *= Gluon({t, (x + n)%Nx, y, z, 1});
            Gluonchain({t, x, y, z, 2}) *= Gluon({t, x, (y + n)%Ny, z, 2});
            Gluonchain({t, x, y, z, 3}) *= Gluon({t, x, y, (z + n)%Nz, 3});
        }
    }
    #pragma omp parallel for reduction(+:W)
    for (int t = 0; t < Nt; ++t)
    {
        int tp {(t + Nmu_end)%Nt};
        for (int x = 0; x < Nx; ++x)
        {
            int xp {(x + Nmu_end)%Nx};
            for (int y = 0; y < Ny; ++y)
            {
                int yp {(y + Nmu_end)%Ny};
                for (int z = 0; z < Nz; ++z)
                {
                    int zp {(z + Nmu_end)%Nz};
                    // W += (Gluonchain({t, x, y, z, 0}) * Gluonchain({tp, x, y, z, 1}) * Gluonchain({t, xp, y, z, 0}).adjoint() * Gluonchain({t, x, y, z, 1}).adjoint()).cast<std::complex<double>>();
                    // W += (Gluonchain({t, x, y, z, 0}) * Gluonchain({tp, x, y, z, 2}) * Gluonchain({t, x, yp, z, 0}).adjoint() * Gluonchain({t, x, y, z, 2}).adjoint()).cast<std::complex<double>>();
                    // W += (Gluonchain({t, x, y, z, 0}) * Gluonchain({tp, x, y, z, 3}) * Gluonchain({t, x, y, zp, 0}).adjoint() * Gluonchain({t, x, y, z, 3}).adjoint()).cast<std::complex<double>>();
                    // W += (Gluonchain({t, x, y, z, 1}) * Gluonchain({t, xp, y, z, 2}) * Gluonchain({t, x, yp, z, 1}).adjoint() * Gluonchain({t, x, y, z, 2}).adjoint()).cast<std::complex<double>>();
                    // W += (Gluonchain({t, x, y, z, 1}) * Gluonchain({t, xp, y, z, 3}) * Gluonchain({t, x, y, zp, 1}).adjoint() * Gluonchain({t, x, y, z, 3}).adjoint()).cast<std::complex<double>>();
                    // W += (Gluonchain({t, x, y, z, 2}) * Gluonchain({t, x, yp, z, 3}) * Gluonchain({t, x, y, zp, 2}).adjoint() * Gluonchain({t, x, y, z, 3}).adjoint()).cast<std::complex<double>>();

                    W += std::real((Gluonchain({t, x, y, z, 0}) * Gluonchain({tp, x, y, z, 1}) * Gluonchain({t, xp, y, z, 0}).adjoint() * Gluonchain({t, x, y, z, 1}).adjoint()).trace());
                    W += std::real((Gluonchain({t, x, y, z, 0}) * Gluonchain({tp, x, y, z, 2}) * Gluonchain({t, x, yp, z, 0}).adjoint() * Gluonchain({t, x, y, z, 2}).adjoint()).trace());
                    W += std::real((Gluonchain({t, x, y, z, 0}) * Gluonchain({tp, x, y, z, 3}) * Gluonchain({t, x, y, zp, 0}).adjoint() * Gluonchain({t, x, y, z, 3}).adjoint()).trace());
                    W += std::real((Gluonchain({t, x, y, z, 1}) * Gluonchain({t, xp, y, z, 2}) * Gluonchain({t, x, yp, z, 1}).adjoint() * Gluonchain({t, x, y, z, 2}).adjoint()).trace());
                    W += std::real((Gluonchain({t, x, y, z, 1}) * Gluonchain({t, xp, y, z, 3}) * Gluonchain({t, x, y, zp, 1}).adjoint() * Gluonchain({t, x, y, z, 3}).adjoint()).trace());
                    W += std::real((Gluonchain({t, x, y, z, 2}) * Gluonchain({t, x, yp, z, 3}) * Gluonchain({t, x, y, zp, 2}).adjoint() * Gluonchain({t, x, y, z, 3}).adjoint()).trace());

                    // W += std::real((Gluonchain({t1, x1, y1, z1, 0}) * Gluonchain({t2, x1, y1, z1, 1}) * Gluonchain({t1, x2, y1, z1, 0}).adjoint() * Gluonchain({t1, x1, y1, z1, 1}).adjoint()).trace());
                    // W += std::real((Gluonchain({t1, x1, y1, z1, 0}) * Gluonchain({t2, x1, y1, z1, 2}) * Gluonchain({t1, x1, y2, z1, 0}).adjoint() * Gluonchain({t1, x1, y1, z1, 2}).adjoint()).trace());
                    // W += std::real((Gluonchain({t1, x1, y1, z1, 0}) * Gluonchain({t2, x1, y1, z1, 3}) * Gluonchain({t1, x1, y1, z2, 0}).adjoint() * Gluonchain({t1, x1, y1, z1, 3}).adjoint()).trace());
                    // W += std::real((Gluonchain({t1, x1, y1, z1, 1}) * Gluonchain({t1, x2, y1, z1, 2}) * Gluonchain({t1, x1, y2, z1, 1}).adjoint() * Gluonchain({t1, x1, y1, z1, 2}).adjoint()).trace());
                    // W += std::real((Gluonchain({t1, x1, y1, z1, 1}) * Gluonchain({t1, x2, y1, z1, 3}) * Gluonchain({t1, x1, y1, z2, 1}).adjoint() * Gluonchain({t1, x1, y1, z1, 3}).adjoint()).trace());
                    // W += std::real((Gluonchain({t1, x1, y1, z1, 2}) * Gluonchain({t1, x1, y2, z1, 3}) * Gluonchain({t1, x1, y1, z2, 2}).adjoint() * Gluonchain({t1, x1, y1, z1, 3}).adjoint()).trace());
                }
            }
        }
    }
    // return 1.0 - std::real(W.trace())/18.0 * full_norm;
    return 1.0 - W/18.0 * full_norm;
}

#endif // LETTUCE_WILSON_LOOP_HPP
