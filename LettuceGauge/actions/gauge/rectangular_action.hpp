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
    template<int stencil_radius_, double c_plaq_, double c_rect_>
    class Rectangular
    {
        // Do not store the gauge field/a reference to the gauge field in the class, since we might want to use the same action for different fields (e.g. during smearing?)
        // Instead, the field is always passed as an external reference
        private:
            double beta;
        public:
            static constexpr int    stencil_radius {stencil_radius_};
            // Coefficients for 1x1 loops (c_plaq) and 1x2/2x1 loops (c_rect)
            static constexpr double c_plaq         {c_plaq_};
            static constexpr double c_rect         {c_rect_};
            //...
            Rectangular(const double beta_in) noexcept :
            beta(beta_in)
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
            double ActionLocal(const link_coord& link, const Matrix_3x3& st_plaq, const Matrix_3x3& st_rect) const noexcept
            {
                return beta * (c_plaq * (1.0 - 1.0/3.0 * std::real((link * st_plaq.adjoint()).trace())) + c_rect * (2.0 - 1.0/3.0 * std::real((link * st_rect.adjoint()).trace())));
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
                        // TODO: Need to implement RectangularLoop function
                        sum_rect += std::real((RectangularLoop<1, 2>(Gluon, {t, x, y, z}, mu, nu)).trace());
                        sum_rect += std::real((RectangularLoop<2, 1>(Gluon, {t, x, y, z}, mu, nu)).trace());
                    }
                }
                return beta * (c_plaq * (6.0 * Gluon.Volume() - 1.0/3.0 * sum_plaq) + c_rect * (12.0 - 1.0/3.0 * sum_rect));
            }

            [[nodiscard]]
            double ActionNormalized(const GaugeField& Gluon) const noexcept
            {
                // TODO: Okay like this, or rewrite this without calling Action()?
                return Action(Gluon) / (6.0 * beta * Gluon.Volume());
            }

            // TODO: Actually write this
            // [[nodiscard]]
            // Matrix_3x3 StaplePlaq(const GaugeField& Gluon, const link_coord& link) const noexcept
            // {
            //     //...
            // }

            // [[nodiscard]]
            // Matrix_3x3 StapleRect(const GaugeField& Gluon, const link_coord& link) const noexcept
            // {
            //     //...
            // }

            // [[nodiscard]]
            // Matrix_3x3 Staple(const GaugeField& Gluon, const link_coord& link) const noexcept
            // {
            //     if constexpr(stencil_radius == 1)
            //     {
            //         return StaplePlaq(Gluon, link);
            //     }
            //     else
            //     {
            //         return StaplePlaq(Gluon, link) + StapleRect(Gluon, link);
            //     }
            // }

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
    using Wilson            = Rectangular<1, 1.0                 ,  0.0>;
    using SymanzikTreeLevel = Rectangular<2, 1.0 + 8.0 * 1.0/12.0, -1.0/12.0>;
    using Iwasaki           = Rectangular<2, 1.0 + 8.0 * 0.331   , -0.331>;
    using DBW2              = Rectangular<2, 1.0 + 8.0 * 1.4088  , -1.4088>;
} // namespace GaugeAction

#endif // LETTUCE_RECTANGULAR_GAUGE_ACTION_HPP
