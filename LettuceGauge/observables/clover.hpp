#ifndef LETTUCE_CLOVER_HPP
#define LETTUCE_CLOVER_HPP

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
// Calculates clover term

void CalculateClover(const GaugeField& Gluon, FullTensor& Clov)
{
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        int tm = (t - 1 + Nt)%Nt;
        int xm = (x - 1 + Nx)%Nx;
        int ym = (y - 1 + Ny)%Ny;
        int zm = (z - 1 + Nz)%Nz;
        int tp = (t + 1)%Nt;
        int xp = (x + 1)%Nx;
        int yp = (y + 1)%Ny;
        int zp = (z + 1)%Nz;

        Clov(t, x, y, z, 0, 0).setZero();
        Clov(t, x, y, z, 0, 1) = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 1}) * Gluon({t, xp, y, z, 0}).adjoint() * Gluon({t, x, y, z, 1}).adjoint()
                               + Gluon({t, x, y, z, 1}) * Gluon({tm, xp, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 1}).adjoint() * Gluon({tm, x, y, z, 0})
                               + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, xm, y, z, 1}).adjoint() * Gluon({tm, xm, y, z, 0}) * Gluon({t, xm, y, z, 1})
                               + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, z, 0}) * Gluon({tp, xm, y, z, 1}) * Gluon({t, x, y, z, 0}).adjoint();
        Clov(t, x, y, z, 1, 0) = Clov(t, x, y, z, 0, 1).adjoint();

        Clov(t, x, y, z, 0, 2) = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 2}) * Gluon({t, x, yp, z, 0}).adjoint() * Gluon({t, x, y, z, 2}).adjoint()
                               + Gluon({t, x, y, z, 2}) * Gluon({tm, x, yp, z, 0}).adjoint() * Gluon({tm, x, y, z, 2}).adjoint() * Gluon({tm, x, y, z, 0})
                               + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, ym, z, 2}).adjoint() * Gluon({tm, x, ym, z, 0}) * Gluon({t, x, ym, z, 2})
                               + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 0}) * Gluon({tp, x, ym, z, 2}) * Gluon({t, x, y, z, 0}).adjoint();
        Clov(t, x, y, z, 2, 0) = Clov(t, x, y, z, 0, 2).adjoint();

        Clov(t, x, y, z, 0, 3) = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 3}) * Gluon({t, x, y, zp, 0}).adjoint() * Gluon({t, x, y, z, 3}).adjoint()
                               + Gluon({t, x, y, z, 3}) * Gluon({tm, x, y, zp, 0}).adjoint() * Gluon({tm, x, y, z, 3}).adjoint() * Gluon({tm, x, y, z, 0})
                               + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, zm, 3}).adjoint() * Gluon({tm, x, y, zm, 0}) * Gluon({t, x, y, zm, 3})
                               + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 0}) * Gluon({tp, x, y, zm, 3}) * Gluon({t, x, y, z, 0}).adjoint();
        Clov(t, x, y, z, 3, 0) = Clov(t, x, y, z, 0, 3).adjoint();

        Clov(t, x, y, z, 1, 1).setZero();
        Clov(t, x, y, z, 1, 2) = Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 2}) * Gluon({t, x, yp, z, 1}).adjoint() * Gluon({t, x, y, z, 2}).adjoint()
                               + Gluon({t, x, y, z, 2}) * Gluon({t, xm, yp, z, 1}).adjoint() * Gluon({t, xm, y, z, 2}).adjoint() * Gluon({t, xm, y, z, 1})
                               + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, ym, z, 2}).adjoint() * Gluon({t, xm, ym, z, 1}) * Gluon({t, x, ym, z, 2})
                               + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 1}) * Gluon({t, xp, ym, z, 2}) * Gluon({t, x, y, z, 1}).adjoint();
        Clov(t, x, y, z, 2, 1) = Clov(t, x, y, z, 1, 2).adjoint();

        Clov(t, x, y, z, 1, 3) = Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 3}) * Gluon({t, x, y, zp, 1}).adjoint() * Gluon({t, x, y, z, 3}).adjoint()
                               + Gluon({t, x, y, z, 3}) * Gluon({t, xm, y, zp, 1}).adjoint() * Gluon({t, xm, y, z, 3}).adjoint() * Gluon({t, xm, y, z, 1})
                               + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, zm, 3}).adjoint() * Gluon({t, xm, y, zm, 1}) * Gluon({t, x, y, zm, 3})
                               + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 1}) * Gluon({t, xp, y, zm, 3}) * Gluon({t, x, y, z, 1}).adjoint();
        Clov(t, x, y, z, 3, 1) = Clov(t, x, y, z, 1, 3).adjoint();

        Clov(t, x, y, z, 2, 2).setZero();
        Clov(t, x, y, z, 2, 3) = Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 3}) * Gluon({t, x, y, zp, 2}).adjoint() * Gluon({t, x, y, z, 3}).adjoint()
                               + Gluon({t, x, y, z, 3}) * Gluon({t, x, ym, zp, 2}).adjoint() * Gluon({t, x, ym, z, 3}).adjoint() * Gluon({t, x, ym, z, 2})
                               + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, zm, 3}).adjoint() * Gluon({t, x, ym, zm, 2}) * Gluon({t, x, y, zm, 3})
                               + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 2}) * Gluon({t, x, yp, zm, 3}) * Gluon({t, x, y, z, 2}).adjoint();
        Clov(t, x, y, z, 3, 2) = Clov(t, x, y, z, 2, 3).adjoint();
        Clov(t, x, y, z, 3, 3).setZero();
    }
}

[[nodiscard]]
Matrix_3x3 CloverDerivativeComponent(const GaugeField& Gluon, const FullTensor& Clover, const site_coord& current_site, const int mu, const int nu, const int rho, const int sigma) noexcept
{
    site_coord site_mup     = Move< 1>(current_site, mu);
    site_coord site_mud     = Move<-1>(current_site, mu);
    site_coord site_nup     = Move< 1>(current_site, nu);
    site_coord site_nud     = Move<-1>(current_site, nu);
    site_coord site_mup_nup = Move< 1>(site_mup, nu);
    site_coord site_mup_nud = Move<-1>(site_mup, nu);

    // TODO: We can probably move the first multiplication outside into the function CloverDerivative(), would go from 6 -> 1 multiplications (although relatively speaking it's not that much)
    return Gluon(current_site, mu) * (Gluon(site_mup, nu)               * Gluon(site_nup, mu).adjoint()     * Gluon(current_site, nu).adjoint() * Clover(current_site, rho, sigma)
                                    + Gluon(site_mup, nu)               * Gluon(site_nup, mu).adjoint()     * Clover(site_nup, rho, sigma)      * Gluon(current_site, nu).adjoint()
                                    + Gluon(site_mup, nu)               * Clover(site_mup_nup, rho, sigma)  * Gluon(site_nup, mu).adjoint()     * Gluon(current_site, nu).adjoint()
                                    + Clover(site_mup, rho, sigma)      * Gluon(site_mup, nu)               * Gluon(site_nup, mu).adjoint()     * Gluon(current_site, nu).adjoint()
                                    - Gluon(site_mup_nud, nu).adjoint() * Gluon(site_nud, mu).adjoint()     * Gluon(site_nud, nu)               * Clover(current_site, rho, sigma)
                                    - Gluon(site_mup_nud, nu).adjoint() * Gluon(site_nud, mu).adjoint()     * Clover(site_nud, rho, sigma)      * Gluon(site_nud, nu)
                                    - Gluon(site_mup_nud, nu).adjoint() * Clover(site_mup_nud, rho, sigma)  * Gluon(site_nud, mu).adjoint()     * Gluon(site_nud, nu)
                                    - Clover(site_mup, rho, sigma)      * Gluon(site_mup_nud, nu).adjoint() * Gluon(site_nud, mu).adjoint()     * Gluon(site_nud, nu));
}

[[nodiscard]]
Matrix_3x3 CloverDerivative(const GaugeField& Gluon, const FullTensor& Clover, const site_coord& current_site, const int mu) noexcept
{
    Matrix_3x3 derivative_component {Matrix_3x3::Zero()};
    // This is basically epsilon_{mu, nu, rho, sigma} manually worked out (not sure if writing a function makes sense?)
    switch(mu)
    {
        case 0:
        {
            derivative_component += CloverDerivativeComponent(Gluon, Clover, current_site, mu, 1, 2, 3);
            derivative_component -= CloverDerivativeComponent(Gluon, Clover, current_site, mu, 1, 3, 2);
            derivative_component -= CloverDerivativeComponent(Gluon, Clover, current_site, mu, 2, 1, 3);
            derivative_component += CloverDerivativeComponent(Gluon, Clover, current_site, mu, 2, 3, 1);
            derivative_component += CloverDerivativeComponent(Gluon, Clover, current_site, mu, 3, 1, 2);
            derivative_component -= CloverDerivativeComponent(Gluon, Clover, current_site, mu, 3, 2, 1);
        }
        break;
        case 1:
        {
            derivative_component -= CloverDerivativeComponent(Gluon, Clover, current_site, mu, 0, 2, 3);
            derivative_component += CloverDerivativeComponent(Gluon, Clover, current_site, mu, 0, 3, 2);
            derivative_component += CloverDerivativeComponent(Gluon, Clover, current_site, mu, 2, 0, 3);
            derivative_component -= CloverDerivativeComponent(Gluon, Clover, current_site, mu, 2, 3, 0);
            derivative_component -= CloverDerivativeComponent(Gluon, Clover, current_site, mu, 3, 0, 2);
            derivative_component += CloverDerivativeComponent(Gluon, Clover, current_site, mu, 3, 2, 0);
        }
        break;
        case 2:
        {
            derivative_component += CloverDerivativeComponent(Gluon, Clover, current_site, mu, 0, 1, 3);
            derivative_component -= CloverDerivativeComponent(Gluon, Clover, current_site, mu, 0, 3, 1);
            derivative_component -= CloverDerivativeComponent(Gluon, Clover, current_site, mu, 1, 0, 3);
            derivative_component += CloverDerivativeComponent(Gluon, Clover, current_site, mu, 1, 3, 0);
            derivative_component += CloverDerivativeComponent(Gluon, Clover, current_site, mu, 3, 0, 1);
            derivative_component -= CloverDerivativeComponent(Gluon, Clover, current_site, mu, 3, 1, 0);
        }
        break;
        case 3:
        {
            derivative_component -= CloverDerivativeComponent(Gluon, Clover, current_site, mu, 0, 1, 2);
            derivative_component += CloverDerivativeComponent(Gluon, Clover, current_site, mu, 0, 2, 1);
            derivative_component += CloverDerivativeComponent(Gluon, Clover, current_site, mu, 1, 0, 2);
            derivative_component -= CloverDerivativeComponent(Gluon, Clover, current_site, mu, 1, 2, 0);
            derivative_component -= CloverDerivativeComponent(Gluon, Clover, current_site, mu, 2, 0, 1);
            derivative_component += CloverDerivativeComponent(Gluon, Clover, current_site, mu, 2, 1, 0);
        }
        break;
    }
    // This is still missing the generators in front
    // Also, this is technically only half of the whole derivative term
    // The remaining half however is simply the adjoint
    // Including the adjoint and simplifying some things, we get two times the real part in the end
    // Without the trace we would get two times the hermitian part, with the trace it's only the real part (?)
    // derivative_component = -static_cast<floatT>(4.0/64.0) * (static_cast<floatT>(1.0/3.0) * derivative_component.trace() * Matrix_3x3::Identity() - derivative_component).real();
    Matrix_3x3 tmp {derivative_component - derivative_component.adjoint()};
    // TODO: I think this is not correct yet, since it includes the prefactor coming from the field strength tensor, but not the prefactor 1/(32 pi^2) from the charge definition
    // return -static_cast<floatT>(1.0/32.0) * (tmp - static_cast<floatT>(1.0/3.0) * tmp.trace() * Matrix_3x3::Identity());
    // This should hopefully be correct
    // return -static_cast<floatT>(1.0/(1024.0 * pi<floatT> * pi<floatT>)) * (tmp - static_cast<floatT>(1.0/3.0) * tmp.trace() * Matrix_3x3::Identity());
    // Turns out it wasn't correct... Comparison with numerical derivatives revealed a missing factor 2 (not sure where it comes from yet)
    return -static_cast<floatT>(1.0/(512.0 * pi<floatT> * pi<floatT>)) * (tmp - static_cast<floatT>(1.0/3.0) * tmp.trace() * Matrix_3x3::Identity());
}

#endif // LETTUCE_CLOVER_HPP
