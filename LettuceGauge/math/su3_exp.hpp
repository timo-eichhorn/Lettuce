#ifndef LETTUCE_SU3_EXP_HPP
#define LETTUCE_SU3_EXP_HPP

// Non-standard library headers
#include "../defines.hpp"
//-----
#include <Eigen/Dense>
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <complex>
//----------------------------------------
// Standard C headers
#include <cmath>

namespace SU3
{
    [[nodiscard]]
    std::complex<floatT> xi_0(const std::complex<floatT> w) noexcept
    {
        if (std::abs(w) <= static_cast<floatT>(0.05))
        {
            std::complex<floatT> w_sq {w * w};
            return static_cast<floatT>(1.0) - static_cast<floatT>(1.0/6.0) * w_sq * (static_cast<floatT>(1.0) - static_cast<floatT>(1.0/20.0) * w_sq * (static_cast<floatT>(1.0) - static_cast<floatT>(1.0/42.0) * w_sq));
        }
        else
        {
            return std::sin(w)/w;
        }
    }

    [[nodiscard]]
    std::complex<floatT> xi_1(const std::complex<floatT> w) noexcept
    {
        return std::cos(w)/(w * w) - std::sin(w)/(w * w * w);
    }

    [[nodiscard]]
    Matrix_SU3 exp(Matrix_3x3 Mat) noexcept
    {
        // Matrix squared (used multiple times below, so compute once)
        Matrix_3x3           Mat2    {Mat * Mat};
        // Determinant and trace of a hermitian matrix are real
        floatT               c0      {static_cast<floatT>(1.0/3.0) * std::real((Mat * Mat2).trace())};
        floatT               c1      {static_cast<floatT>(0.5) * std::real(Mat2.trace())};
        std::complex<floatT> c0_max  {static_cast<floatT>(2.0) * std::pow(c1 / static_cast<floatT>(3.0), static_cast<floatT>(1.5))};
        std::complex<floatT> theta   {std::acos(c0/c0_max)};
        std::complex<floatT> u       {std::sqrt(static_cast<floatT>(1.0/3.0) * c1) * std::cos(static_cast<floatT>(1.0/3.0) * theta)};
        std::complex<floatT> w       {std::sqrt(c1) * std::sin(static_cast<floatT>(1.0/3.0) * theta)};
        // Auxiliary variables depending on u that get used more than once
        std::complex<floatT> u2      {u * u};
        std::complex<floatT> exp_iu  {std::exp(-i<floatT> * u)};
        std::complex<floatT> exp_2iu {std::exp(static_cast<floatT>(2.0) * i<floatT> * u)};
        // Auxiliary variables depending on w that get used more than once
        std::complex<floatT> w2      {w * w};
        std::complex<floatT> cosw    {std::cos(w)};
        std::complex<floatT> i_xi0   {i<floatT> * xi_0(w)};
        // Denominator of f_1, f_2, f_3
        std::complex<floatT> denom   {static_cast<floatT>(1.0) / (static_cast<floatT>(9.0) * u2 - w2)};
        // h_0, h_1, h_2 functions to be used during calculation of f_1, f_2, f_3
        // TODO: Numerically problematic if w -> 3u -> sqrt(3)/2 as c0 -> -c0_max?
        // Can be circumvented by using symmetry relation of f_j, but is that really necessary here?
        // If so, we only want to check once if c0 is negative
        std::complex<floatT> h0      {(u2 - w2) * exp_2iu + exp_iu * (static_cast<floatT>(8.0) * u2 * cosw + static_cast<floatT>(2.0) * u * i_xi0 * (static_cast<floatT>(3.0) * u2 + w2))};
        std::complex<floatT> h1      {static_cast<floatT>(2.0) * u * exp_2iu - exp_iu * (static_cast<floatT>(2.0) * u * cosw - (static_cast<floatT>(3.0) * u2 - w2) * i_xi0)};
        std::complex<floatT> h2      {exp_2iu - exp_iu * (cosw + static_cast<floatT>(3.0) * u * i_xi0)};
        // Final return value
        return denom * (h0 * Matrix_3x3::Identity() + h1 * Mat + h2 * Mat2);
    }
}

#endif // LETTUCE_SU3_EXP_HPP
