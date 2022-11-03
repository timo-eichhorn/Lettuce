#ifndef LETTUCE_RECTANGULAR_GAUGE_ACTION_HPP
#define LETTUCE_RECTANGULAR_GAUGE_ACTION_HPP

// Non-standard library headers
#include "../../defines.hpp"
#include "../../coords.hpp"
#include "../../observables/wilson_loop.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <complex>
//----------------------------------------
// Standard C headers
// ...

namespace GaugeAction
{
    // TODO: Unfortunately, only GCC currently supports floating-point types as template parameters, both Clang and the Intel LLVM compiler will complain
    // template<int stencil_radius_, double c_plaq_, double c_rect_>
    template<int stencil_radius_>
    class Rectangular
    {
        // Do not store the gauge field/a reference to the gauge field in the class, since we might want to use the same action for different fields (e.g. during smearing?)
        // Instead, the field is always passed as an external reference
        private:
            double beta;
        public:
            static constexpr int    stencil_radius {stencil_radius_};
            static_assert(stencil_radius_ > 0 and stencil_radius_ <= 2, "Invalid stencil_radius!");
            // Coefficients for 1x1 loops (c_plaq) and 1x2/2x1 loops (c_rect)
            // static constexpr double c_plaq         {c_plaq_};
            // static constexpr double c_rect         {c_rect_};
            const double c_plaq;
            const double c_rect;
            //...
            Rectangular(const double beta_in, const double c_plaq_in, const double c_rect_in) noexcept :
            beta(beta_in), c_plaq(c_plaq_in), c_rect(c_rect_in)
            {}

            void SetBeta(const double beta_in) noexcept
            {
                beta = beta_in;
            }

            [[nodiscard]]
            double GetBeta() const noexcept
            {
                return beta;
            }

            [[nodiscard]]
            double ActionLocal(const Matrix_SU3& U, const Matrix_3x3& st_plaq, const Matrix_3x3& st_rect) const noexcept
            {
                return beta * (c_plaq * (1.0 - 1.0/3.0 * std::real((U * st_plaq.adjoint()).trace())) + c_rect * (2.0 - 1.0/3.0 * std::real((U * st_rect.adjoint()).trace())));
            }

            [[nodiscard]]
            double Action(const GaugeField& Gluon) const noexcept
            {
                double sum_plaq {0.0};
                double sum_rect {0.0};

                #pragma omp parallel for reduction(+:sum_plaq,sum_rect)
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int nu = 1; nu < 4; ++nu)
                {
                    for (int mu = 0; mu < nu; ++mu)
                    {
                        // Plaquette contributions
                        sum_plaq += std::real(Plaquette(Gluon, {t, x, y, z}, mu, nu).trace());
                        // Rectangle contributions
                        if constexpr(stencil_radius == 2)
                        {
                            sum_rect += std::real((RectangularLoop<1, 2>(Gluon, {t, x, y, z}, mu, nu)).trace());
                            sum_rect += std::real((RectangularLoop<2, 1>(Gluon, {t, x, y, z}, mu, nu)).trace());
                        }
                    }
                }
                if constexpr(stencil_radius == 1)
                {
                    return beta * c_plaq * (6.0 * Gluon.Volume() - 1.0/3.0 * sum_plaq);
                }
                else
                {
                    return beta * (c_plaq * (6.0 * Gluon.Volume() - 1.0/3.0 * sum_plaq) + c_rect * (12.0 * Gluon.Volume() - 1.0/3.0 * sum_rect));
                }
            }

            [[nodiscard]]
            double ActionNormalized(const GaugeField& Gluon) const noexcept
            {
                return Action(Gluon) / (6.0 * beta * Gluon.Volume());
            }

            // TODO: For now, the plaquette staple is written out explicitly, since the reduced index calculations lead to slightly better performance
            [[nodiscard]]
            Matrix_3x3 StaplePlaq(const GaugeField& Gluon, const link_coord& current_link) const noexcept
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

            [[nodiscard]]
            Matrix_3x3 StapleRect(const GaugeField& U, const site_coord& current_site, const int mu) const noexcept
            {
                Matrix_3x3 st;
                for (int nu_offset = 1; nu_offset < 4; ++nu_offset)
                {
                    int nu {(mu + nu_offset) % 4};
                    // First term, orthogonal to link
                    // Right half
                    site_coord site_nup      {Move< 1>(current_site, nu)};
                    site_coord site_nupp     {Move< 1>(site_nup    , nu)};
                    site_coord site_mup_nup  {Move< 1>(site_nup    , mu)};
                    site_coord site_mup      {Move< 1>(current_site, mu)};
                    // Left half
                    site_coord site_nud      {Move<-1>(current_site, nu)};
                    site_coord site_nudd     {Move<-1>(site_nud    , nu)};
                    site_coord site_mup_nudd {Move< 1>(site_nudd   , mu)};
                    site_coord site_mup_nud  {Move< 1>(site_nud    , mu)};
                    st.noalias() += U(current_site, nu)           * U(site_nup , nu)           * U(site_nupp, mu) * U(site_mup_nup , nu).adjoint() * U(site_mup    , nu).adjoint()
                                  + U(site_nud    , nu).adjoint() * U(site_nudd, nu).adjoint() * U(site_nudd, mu) * U(site_mup_nudd, nu)           * U(site_mup_nud, nu);
                    //-----
                    // Second term, same direction as link (staple originating from current_site)
                    // Right half
                    site_coord site_mupp     {Move< 1>(site_mup    , mu)};
                    // Left half
                    site_coord site_mupp_nud {Move< 1>(site_mup_nud, mu)};
                    st.noalias() += (U(current_site, nu)           * U(site_nup, mu) * U(site_mup_nup, mu) * U(site_mupp    , nu).adjoint()
                                   + U(site_nud    , nu).adjoint() * U(site_nud, mu) * U(site_mup_nud, mu) * U(site_mupp_nud, nu)) * U(site_mup, mu).adjoint();
                    //-----
                    // Third term, same direction as link (staple originating from site_mud)
                    // Right half
                    site_coord site_mud      {Move<-1>(current_site, mu)};
                    site_coord site_mud_nup  {Move< 1>(site_mud    , nu)};
                    // Left half
                    site_coord site_mud_nud  {Move<-1>(site_mud    , nu)};
                    st.noalias() += U(site_mud, mu).adjoint() * (U(site_mud    , nu)           * U(site_mud_nup, mu) * U(site_nup, mu) * U(site_mup    , nu).adjoint()
                                                               + U(site_mud_nud, nu).adjoint() * U(site_mud_nud, mu) * U(site_nud, mu) * U(site_mup_nud, nu));
                }
                return st;
            }

            [[nodiscard]]
            Matrix_3x3 StapleRect(const GaugeField& U, const link_coord& current_link) const noexcept
            {
                auto [t, x, y, z, mu] = current_link;
                return StapleRect(U, {t, x, y, z}, mu);
            }

            [[nodiscard]]
            Matrix_3x3 Staple(const GaugeField& Gluon, const link_coord& current_link) const noexcept
            {
                if constexpr(stencil_radius == 1)
                {
                    return StaplePlaq(Gluon, current_link);
                }
                else
                {
                    return c_plaq * StaplePlaq(Gluon, current_link) + c_rect * StapleRect(Gluon, current_link);
                }
            }

            [[nodiscard]]
            double Local(const Matrix_SU3& U, const Matrix_3x3& st) noexcept
            {
                return beta * (c_plaq + 2.0 * c_rect - 1.0/3.0 * std::real((U * st.adjoint()).trace()));
            }
    };

    // Template specialization for Wilson gauge action?
    // template<>
    // [[nodiscard]]
    // double Rectangular<1, 1.0, 0.0>::ActionLocal(const link_coord& link, const Matrix_3x3& st_plaq) const noexcept {...;}

    // template<>
    // [[nodiscard]]
    // double Rectangular<1, 1.0, 0.0>::Action(const GaugeField& Gluon) const noexcept {...;}

    // // TODO: Do we need to specialize this, or will the normal definition work automatically once we specialize Action()?
    // template<>
    // [[nodiscard]]
    // double Rectangular<1, 1.0, 0.0>::ActionNormalized(const GaugeField& Gluon) const noexcept {...;}

    // template<>
    // [[nodiscard]]
    // Matrix_3x3 Staple(const GaugeField& Gluon, const link_coord& link) const noexcept {...;}

    // Some commonly used improved gauge actions:
    // For all actions we define c_plaq = 1 - 8 * c_rect
    // -----
    // Wilson               : c_rect =  0
    // Symanzik (Tree level): c_rect = -1/12
    // Iwasaki              : c_rect = -0.331
    // DBW2                 : c_rect = -1.4088
    // -----
    // using Wilson            = Rectangular<1, 1.0                 ,  0.0>;
    // using SymanzikTreeLevel = Rectangular<2, 1.0 + 8.0 * 1.0/12.0, -1.0/12.0>;
    // using Iwasaki           = Rectangular<2, 1.0 + 8.0 * 0.331   , -0.331>;
    // using DBW2              = Rectangular<2, 1.0 + 8.0 * 1.4088  , -1.4088>;
    Rectangular<1>  WilsonAction      (beta, 1.0                 ,  0.0);
    Rectangular<2>  LüscherWeiszAction(beta, 1.0 + 8.0 * 1.0/12.0, -1.0/12.0);
    Rectangular<2>  IwasakiAction     (beta, 1.0 + 8.0 * 0.331   , -0.331);
    Rectangular<2>  DBW2Action        (beta, 1.0 + 8.0 * 1.4088  , -1.4088);
} // namespace GaugeAction

#endif // LETTUCE_RECTANGULAR_GAUGE_ACTION_HPP
