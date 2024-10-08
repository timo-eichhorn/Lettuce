#ifndef LETTUCE_HEATBATH_HPP
#define LETTUCE_HEATBATH_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../math/su2.hpp"
#include "../math/su3.hpp"
//-----
#include <Eigen/Dense>
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <random>
//----------------------------------------
// Standard C headers
#include <cmath>

//+---------------------------------------------------------------------------------+
//| This file provides a functor implementing a pseudo heat bath update for SU(3)   |
//| gauge theory, based on a Cabibbo-Marinari decomposition into SU(2) subgroups.   |
//| The original heat bath update is described in PhysRevD.21.2308, and the         |
//| decomposition into SU(2) subgroups was proposed in PhysLettB.119.387.           |
//| There are two papers proposing an improvement to the sampling procedure compared|
//| to the original Creutz paper which are used here, namely PhysLettB.143.459 and  |
//| PhysLettB.156.393.                                                              |
//+---------------------------------------------------------------------------------+

template<typename ActionT, typename prngT>
struct HeatbathKernel
{
    private:
        GaugeField& U;
        // TODO: Check if the stencil_radius of the Action is larger than 1 to prevent incorrect masking/parallelization
        ActionT&    Action;
        prngT&      prng;
        // TODO: Include this as parameters in constructor
        int         N_col {3};
        floatT      prefactor {static_cast<floatT>(N_col) / beta};
        int         max_iteration;

        // TODO: Provide two functions to generate SU(2) links (original Creutz algorithm which might be more efficient for very low beta, as well as already existing
        //       Fabricius-Haan/Kennedy-Pendleton algorithm)?
        // SU(2) heat bath which can be used in combination with a Cabibbo-Marinari decomposition to obtain a SU(N) pseudo heat bath algorithm
        [[nodiscard]]
        SU2_comp<floatT> HeatbathSU2(const SU2_comp<floatT>& A, const floatT prefactor, const link_coord& current_link, const int max_iteration) const noexcept
        {
            // Determinant of staple as norm to project staple back to SU(2)
            floatT           a_norm {static_cast<floatT>(1.0) / A.det_sqrt()};
            SU2_comp<floatT> V      {a_norm * A};
            floatT           r1, r2, r3, x1, x2, x3, lambda_sq, r0;
            int iteration_count {0};
            do
            {
                // Generate random number lambda_sq following a polynomially modified Gaussian distribution (cf. Gattringer & Lang (4.43))
                r1 = static_cast<floatT>(1.0) - prng.UniformReal(current_link);
                x1 = std::log(r1);
                r2 = static_cast<floatT>(1.0) - prng.UniformReal(current_link);
                x2 = std::cos(static_cast<floatT>(2.0) * pi<floatT> * r2);
                r3 = static_cast<floatT>(1.0) - prng.UniformReal(current_link);
                x3 = std::log(r3);
                // Factor 0.25 to get the correct total prefactor
                // For SU(2), the prefactor is 0.5 / beta
                // For SU(3), the prefactor is 0.75 / beta
                // floatT lambda_sq {static_cast<floatT>(-0.25 * prefactor * a_norm) * (x1 + x2 * x2 * x3)};
                lambda_sq = static_cast<floatT>(-0.25 * prefactor * a_norm) * (x1 + x2 * x2 * x3);
                //-----
                // Correct for factor sqrt(1 - lambda_sq) in probability distribution via accept-reject step
                // floatT r0 {prng.UniformReal(current_link)};
                r0 = prng.UniformReal(current_link);
                // After the maximum number of iterations is reached stop and return the unit matrix, i.e., don't update the link
                if (iteration_count > max_iteration)
                {
                    return {1.0, 0.0};
                }
                ++iteration_count;
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
            //       (-1, 1]. Therefore, do something like this: std::uniform_real_distribution<floatT> distribution_uniform(-1.0, std::nextafter(1.0, 2.0))
            //       floatT r1 {static_cast<floatT>(1.0) - static_cast<floatT>(2.0) * prng.UniformReal(current_link)};
            // Random number in interval [0, 1)
            floatT phi {prng.UniformReal(current_link)};
            // Random number in interval (-1, 1]
            floatT cos_theta {static_cast<floatT>(1.0) - static_cast<floatT>(2.0) * prng.UniformReal(current_link)};
            floatT vec_norm {abs_x * std::sqrt(static_cast<floatT>(1.0) - cos_theta * cos_theta)};
            //-----
            x1 = vec_norm * std::cos(static_cast<floatT>(2.0) * pi<floatT> * phi);
            x2 = vec_norm * std::sin(static_cast<floatT>(2.0) * pi<floatT> * phi);
            x3 = abs_x * cos_theta;
            //-----
            return {SU2_comp<floatT> {std::complex<floatT> (x0, x1), std::complex<floatT> (x2, x3)} * V.adjoint()};
        }
    public:
        explicit HeatbathKernel(GaugeField& U_in, ActionT& Action_in, prngT& prng_in, const int max_iteration_in = 10) noexcept :
        U(U_in), Action(Action_in), prng(prng_in), max_iteration(max_iteration_in)
        {}

        void operator()(const link_coord& current_link) const noexcept
        {
            SU2_comp<floatT> subblock;
            // Note: Our staple definition corresponds to the daggered staple in Gattringer & Lang, therefore use adjoint
            Matrix_3x3       st_adj       {(Action.Staple(U, current_link)).adjoint()};
            //-----
            // Update (0, 1) subgroup
            subblock        = Extract01<floatT>(U(current_link) * st_adj);
            U(current_link) = Embed01(HeatbathSU2(subblock, prefactor, current_link, max_iteration)) * U(current_link);
            //-----
            // Update (0, 2) subgroup
            subblock        = Extract02<floatT>(U(current_link) * st_adj);
            U(current_link) = Embed02(HeatbathSU2(subblock, prefactor, current_link, max_iteration)) * U(current_link);
            //-----
            // Update (1, 2) subgroup
            subblock        = Extract12<floatT>(U(current_link) * st_adj);
            U(current_link) = Embed12(HeatbathSU2(subblock, prefactor, current_link, max_iteration)) * U(current_link);
            //-----
            SU3::Projection::GramSchmidt(U(current_link));
        }
};

// TODO: Implement this?
// template<typename ActionT, typename prngT>
// struct HeatbathPietarinenKernel
// {
//     private:
//         GaugeField& U;
//         prngT&      prng;
//     public:
//         explicit HeatbathPietarinenKernel(GaugeField& U_in, prngT& prng_in) noexcept :
//         U(U_in), prng(prng_in)
//         {}

//         void operator()(const link_coord& current_link) const noexcept
//         {
//             int iteration_count {0};
//             do
//             {
//                 floatT alpha1 {prng.UniformReal(current_link)};
//                 floatT alpha2 {prng.UniformReal(current_link)};
//                 floatT alpha3 {prng.UniformReal(current_link)};
//                 floatT alpha4 {prng.UniformReal(current_link)};
//                 floatT alpha5 {prng.UniformReal(current_link)};
//                 floatT alpha6 {prng.UniformReal(current_link)};
//                 floatT alpha7 {prng.UniformReal(current_link)};
//                 floatT alpha8 {prng.UniformReal(current_link)};
//                 //...
//                 if (iteration_count > max_iteration)
//                 {
//                     return {1.0, 0.0};
//                 }
//                 ++iteration_count;
//             }
//             while (r0 * r0 + lambda_sq >= static_cast<floatT>(1.0));

//         }
// };

#endif // LETTUCE_HEATBATH_HPP
