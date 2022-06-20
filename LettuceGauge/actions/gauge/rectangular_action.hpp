#ifndef LETTUCE_RECTANGULAR_GAUGE_ACTION_HPP
#define LETTUCE_RECTANGULAR_GAUGE_ACTION_HPP

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
            double ActionLocal(const GaugeField& Gluon) noexcept
            {
                return;
            }

            [[nodiscard]]
            double Action(const GaugeField& Gluon) noexcept
            {
                double sum_plaquette {0.0};
                double sum_rectangle {0.0};

                #pragma omp parallel for reduction(+:sum_plaquette,sum_rectangle)
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
                        sum_rect += std::real((RectangularLoop<1,2>(Gluon, {t, x, y, z}, mu, nu)).trace());
                        sum_rect += std::real((RectangularLoop<2,1>(Gluon, {t, x, y, z}, mu, nu)).trace());
                    }
                }
                return beta * (6.0 * Nt * Nx * Ny * Nz - 1.0/3.0 * S);
            }

            [[nodiscard]]
            double ActionNormalized() noexcept
            {
                return;
            }

            [[nodiscard]]
            Matrix_3x3 Staple(const link_coord& link) noexcept
            {
                return;
            }
    };

    // Some commonly used improved gauge actions:
    // c_plaq = 1 - 8 * c_rect
    // Symanzik (Tree level) c_rect = -1/12
    // Iwasaki:              c_rect = -0.331
    // DBW2   :              c_rect = -1.4088
    using SymanzikTreeLevel = Rectangular<2, 1.0 + 8.0 * 1.0/12.0, -1.0/12.0>;
    using Iwasaki           = Rectangular<2, 1.0 + 8.0 * 0.331   , -0.331>;
    using DBW2              = Rectangular<2, 1.0 + 8.0 * 1.4088  , -1.4088>;
}

#endif // LETTUCE_RECTANGULAR_GAUGE_ACTION_HPP
