#ifndef LETTUCE_HEATBATH_HPP
#define LETTUCE_HEATBATH_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../actions/gauge/wilson_action.hpp"
#include "../math/su2.hpp"
#include "../math/su3.hpp"
//-----
#include <Eigen/Dense>
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <random>
//----------------------------------------
// Standard C headers
#include <cmath>

template<typename floatT>
struct HeatbathKernel
{
    private:
        GaugeField& Gluon;
        std::uniform_real_distribution<floatT>& distribution_uniform;
        // TODO: Include this as parameters in constructor
        int N_col {3};
        floatT prefactor {static_cast<floatT>(N_col) / beta};
        int max_iteration;
        SU2_comp<floatT> HeatbathSU2(const SU2_comp<floatT>& A, const floatT prefactor, std::uniform_real_distribution<floatT>& distribution_uniform, const int max_iteration) const noexcept
        {
            // Determinant of staple as norm to project staple back to SU(2)
            floatT a_norm {static_cast<floatT>(1.0) / std::sqrt(A.det_sq())};
            SU2_comp<floatT> V {a_norm * A};
            SU2_comp<floatT> mat_su2;
            floatT r1, r2, r3, x1, x2, x3, lambda_sq, r0;
            int count {0};
            do
            {
                // Generate random number lambda_sq following a polynomially modified Gaussian distribution (cf. Gattringer & Lang (4.43))
                r1 = static_cast<floatT>(1.0) - distribution_uniform(prng_vector[omp_get_thread_num()]);
                x1 = std::log(r1);
                r2 = static_cast<floatT>(1.0) - distribution_uniform(prng_vector[omp_get_thread_num()]);
                x2 = std::cos(static_cast<floatT>(2.0) * pi<floatT> * r2);
                r3 = static_cast<floatT>(1.0) - distribution_uniform(prng_vector[omp_get_thread_num()]);
                x3 = std::log(r3);
                // Factor 0.25, so for N_col = 2 we get a factor 0.5, while for N_col = 3 we get a factor 0.75
                // floatT lambda_sq {static_cast<floatT>(-0.25 * prefactor * a_norm) * (x1 + x2 * x2 * x3)};
                lambda_sq = static_cast<floatT>(-0.25 * prefactor * a_norm) * (x1 + x2 * x2 * x3);
                //-----
                // Correct for factor sqrt(1 - lambda_sq) in probability distribution via accept-reject step
                // floatT r0 {distribution_uniform(prng_vector[omp_get_thread_num()])};
                r0 = distribution_uniform(prng_vector[omp_get_thread_num()]);
                if (count > max_iteration)
                {
                    return {1.0, 0.0};
                }
                ++count;
            }
            while (r0 * r0 + lambda_sq >= static_cast<floatT>(1.0));

            // Calculate zeroth coefficient of our SU(2) matrix in quaternionic representation
            floatT x0 {static_cast<floatT>(1.0) - static_cast<floatT>(2.0) * lambda_sq};
            // Calculate absolute value of our random vector
            floatT abs_x {std::sqrt(static_cast<floatT>(1.0) - x0 * x0)};
            // Generate angular variables, i.e., random vector with length abs_x (simply generating three uniformly distributed values in the unit range and
            // normalizing them does not work, since the resulting distribution is not uniform, but biased against vectors close to the coordinate axes)
            // Instead, we generate a random vector with length abs_x in spherical coordinates
            // Since the functional determinant contains a factor sin(theta), directly generating the coordinates does not give a uniform distribution
            // Therefore, we generate cos(theta) in [-1, 1] using sqrt(1 - rand^2)
            // TODO: We want a random number in the closed interval [-1, 1], but the standard distribution only covers the half open interval [-1, 1) or
            // (-1, 1]. Therefore, do something like this: std::uniform_real_distribution<floatT> distribution_uniform(-1.0, std::nextafter(1.0, 2.0))
            // floatT r1 {static_cast<floatT>(1.0) - static_cast<floatT>(2.0) * distribution_uniform(prng_vector[omp_get_thread_num()])};
            // Random number in interval [0, 1)
            floatT phi {distribution_uniform(prng_vector[omp_get_thread_num()])};
            // Random number in interval (-1, 1]
            floatT cos_theta {static_cast<floatT>(1.0) - static_cast<floatT>(2.0) * distribution_uniform(prng_vector[omp_get_thread_num()])};
            floatT vec_norm {abs_x * std::sqrt(static_cast<floatT>(1.0) - cos_theta * cos_theta)};
            //-----
            x1 = vec_norm * std::cos(static_cast<floatT>(2.0) * pi<floatT> * phi);
            x2 = vec_norm * std::sin(static_cast<floatT>(2.0) * pi<floatT> * phi);
            x3 = abs_x * cos_theta;
            //-----
            return {SU2_comp<floatT> {std::complex<floatT> (x0, x1), std::complex<floatT> (x2, x3)} * V.adjoint()};
        }
    public:
        explicit HeatbathKernel(GaugeField& Gluon_in, std::uniform_real_distribution<floatT>& distribution_uniform_in, const int max_iteration_in = 10) noexcept :
        Gluon(Gluon_in), distribution_uniform(distribution_uniform_in), max_iteration(max_iteration_in)
        {}

        void operator()(const int t, const int x, const int y, const int z, const int mu) const noexcept
        {
            // For SU(2), the prefactor is 0.5 / beta
            // For SU(3), the prefactor is 0.75 / beta
            // floatT prefactor {static_cast<floatT>(0.75) / beta}; // N_c/(2 * 2)
            // int N_col {3};
            // floatT prefactor {static_cast<floatT>(N_col) / beta};
            Matrix_3x3 W;
            SU2_comp<floatT> subblock;
            // Note: Our staple definition corresponds to the daggered staple in Gattringer & Lang, therefore use adjoint
            Matrix_3x3 st_adj {(WilsonAction::Staple(Gluon, t, x, y, z, mu)).adjoint()};
            //-----
            // Update (0, 1) subgroup
            // W = Gluon[t][x][y][z][mu] * st_adj;
            subblock = Extract01<floatT>(Gluon(t, x, y, z, mu) * st_adj);
            Gluon(t, x, y, z, mu) = Embed01(HeatbathSU2(subblock, prefactor, distribution_uniform, max_iteration)) * Gluon(t, x, y, z, mu);
            //-----
            // Update (0, 2) subgroup
            // W = Gluon[t][x][y][z][mu] * st_adj;
            subblock = Extract02<floatT>(Gluon(t, x, y, z, mu) * st_adj);
            Gluon(t, x, y, z, mu) = Embed02(HeatbathSU2(subblock, prefactor, distribution_uniform, max_iteration)) * Gluon(t, x, y, z, mu);
            //-----
            // Update (1, 2) subgroup
            // W = Gluon[t][x][y][z][mu] * st_adj;
            subblock = Extract12<floatT>(Gluon(t, x, y, z, mu) * st_adj);
            Gluon(t, x, y, z, mu) = Embed12(HeatbathSU2(subblock, prefactor, distribution_uniform, max_iteration)) * Gluon(t, x, y, z, mu);
            //-----
            // Project link to SU(3)
            SU3::Projection::GramSchmidt(Gluon(t, x, y, z, mu));
        }
        void operator()(const link_coord& current_link) const noexcept
        {
            // For SU(2), the prefactor is 0.5 / beta
            // For SU(3), the prefactor is 0.75 / beta
            // floatT prefactor {static_cast<floatT>(0.75) / beta}; // N_c/(2 * 2)
            // int N_col {3};
            // floatT prefactor {static_cast<floatT>(N_col) / beta};
            Matrix_3x3 W;
            SU2_comp<floatT> subblock;
            // Note: Our staple definition corresponds to the daggered staple in Gattringer & Lang, therefore use adjoint
            Matrix_3x3 st_adj {(WilsonAction::Staple(Gluon, current_link.t, current_link.x, current_link.y, current_link.z, current_link.mu)).adjoint()};
            //-----
            // Update (0, 1) subgroup
            // W = Gluon[t][x][y][z][mu] * st_adj;
            subblock = Extract01<floatT>(Gluon(current_link) * st_adj);
            Gluon(current_link) = Embed01(HeatbathSU2(subblock, prefactor, distribution_uniform, max_iteration)) * Gluon(current_link);
            //-----
            // Update (0, 2) subgroup
            // W = Gluon[t][x][y][z][mu] * st_adj;
            subblock = Extract02<floatT>(Gluon(current_link) * st_adj);
            Gluon(current_link) = Embed02(HeatbathSU2(subblock, prefactor, distribution_uniform, max_iteration)) * Gluon(current_link);
            //-----
            // Update (1, 2) subgroup
            // W = Gluon[t][x][y][z][mu] * st_adj;
            subblock = Extract12<floatT>(Gluon(current_link) * st_adj);
            Gluon(current_link) = Embed12(HeatbathSU2(subblock, prefactor, distribution_uniform, max_iteration)) * Gluon(current_link);
            //-----
            // Project link to SU(3)
            SU3::Projection::GramSchmidt(Gluon(current_link));
        }
};

#endif // LETTUCE_HEATBATH_HPP
