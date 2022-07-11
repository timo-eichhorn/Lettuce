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
    floatT return_xi_0(const floatT w) noexcept
    {
        if (std::abs(w) <= static_cast<floatT>(0.05))
        {
            floatT w2 {w * w};
            return static_cast<floatT>(1.0) - static_cast<floatT>(1.0/6.0) * w2 * (static_cast<floatT>(1.0) - static_cast<floatT>(1.0/20.0) * w2 * (static_cast<floatT>(1.0) - static_cast<floatT>(1.0/42.0) * w2));
        }
        else
        {
            return std::sin(w)/w;
        }
    }

    [[nodiscard]]
    floatT return_xi_1(const floatT w) noexcept
    {
        // return std::cos(w)/(w * w) - std::sin(w)/(w * w * w);
        return static_cast<floatT>(1.0)/(w * w) * (std::cos(w) - std::sin(w)/w);
    }

    struct uDerivedConstants
    {
        floatT               u;
        floatT               u2;
        std::complex<floatT> exp_miu;
        std::complex<floatT> exp_2iu;
        // Calculate u derived constants from c1 and theta
        uDerivedConstants(const floatT c1, const floatT theta) noexcept :
        u       (std::sqrt(static_cast<floatT>(1.0/3.0) * c1) * std::cos(static_cast<floatT>(1.0/3.0) * theta)),
        u2      (u * u),
        exp_miu (std::exp(-i<floatT> * u)),
        exp_2iu (std::exp(static_cast<floatT>(2.0) * i<floatT> * u))
        {}
    };

    struct wDerivedConstants
    {
        floatT               w;
        floatT               w2;
        std::complex<floatT> cosw;
        std::complex<floatT> i_xi0;
        // Calculate w derived constants from c1 and theta
        wDerivedConstants(const floatT c1, const floatT theta) noexcept :
        w     (std::sqrt(c1) * std::sin(static_cast<floatT>(1.0/3.0) * theta)),
        w2    (w * w),
        cosw  (std::cos(w)),
        i_xi0 (i<floatT> * return_xi_0(w))
        {}
    };

    struct rConstants
    {
        std::complex<floatT> r1_0;
        std::complex<floatT> r1_1;
        std::complex<floatT> r1_2;
        std::complex<floatT> r2_0;
        std::complex<floatT> r2_1;
        std::complex<floatT> r2_2;
        //-----
        // TODO: Add xi_0 (without factor i) and xi_1 to wDerivedConstants? We only need it for the derivative of exp, not for exp itself, so calculating xi_1 there is unnecessary
        rConstants(const uDerivedConstants& u_derived, const wDerivedConstants& w_derived, const std::complex<floatT> xi_0, const std::complex<floatT> xi_1) noexcept :
        r1_0 (static_cast<floatT>(2.0) * (u_derived.u + i<floatT> * (u_derived.u2 - w_derived.w2)) * u_derived.exp_2iu
            + static_cast<floatT>(2.0) * u_derived.exp_miu * (static_cast<floatT>(4.0) * u_derived.u * (static_cast<floatT>(2.0) - i<floatT> * u_derived.u) * w_derived.cosw
            + w_derived.i_xi0 * (static_cast<floatT>(9.0) * u_derived.u2 + w_derived.w2 - i<floatT> * u_derived.u * (static_cast<floatT>(3.0) * u_derived.u2 + w_derived.w2)))),
        r1_1 (static_cast<floatT>(2.0) * (static_cast<floatT>(1.0) + static_cast<floatT>(2.0) * i<floatT> * u_derived.u) * u_derived.exp_2iu + u_derived.exp_miu * (-static_cast<floatT>(2.0) * (static_cast<floatT>(1.0) - i<floatT> * u_derived.u) * w_derived.cosw
            + w_derived.i_xi0 * (static_cast<floatT>(6.0) * u_derived.u + i<floatT> * (w_derived.w2 - static_cast<floatT>(3.0) * u_derived.u2)))),
        r1_2 (static_cast<floatT>(2.0) * i<floatT> * u_derived.exp_2iu + i<floatT> * u_derived.exp_miu * (w_derived.cosw - static_cast<floatT>(3.0) * (static_cast<floatT>(1.0) - i<floatT> * u_derived.u) * xi_0)),
        r2_0 (-static_cast<floatT>(2.0) * u_derived.exp_2iu + static_cast<floatT>(2.0) * i<floatT> * u_derived.u * u_derived.exp_miu * (w_derived.cosw + (static_cast<floatT>(1.0) + static_cast<floatT>(4.0) * i<floatT> * u_derived.u) * xi_0 + static_cast<floatT>(3.0) * u_derived.u2 * xi_1)),
        r2_1 (-i<floatT> * u_derived.exp_miu * (w_derived.cosw + (static_cast<floatT>(1.0) + static_cast<floatT>(2.0) * i<floatT> * u_derived.u) * xi_0 - static_cast<floatT>(3.0) * u_derived.u2 * xi_1)),
        r2_2 (u_derived.exp_miu * (xi_0 - static_cast<floatT>(3.0) * i<floatT> * u_derived.u * xi_1))
        {}
    };

    struct bConstants
    {
        std::complex<floatT> b_denom;
        std::complex<floatT> b_10;
        std::complex<floatT> b_11;
        std::complex<floatT> b_12;
        std::complex<floatT> b_20;
        std::complex<floatT> b_21;
        std::complex<floatT> b_22;
        //-----
        // TODO: Encapsulate f_i (and h_i?) in a struct?
        bConstants(const uDerivedConstants& u_derived, const wDerivedConstants& w_derived, const rConstants& r_consts, const std::complex<floatT> f0, const std::complex<floatT> f1, const std::complex<floatT> f2) noexcept :
        b_denom (static_cast<floatT>(0.5) / ((static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2) * (static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2))),
        b_10    (b_denom * (static_cast<floatT>(2.0) * u_derived.u * r_consts.r1_0 + (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * r_consts.r2_0 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u_derived.u2 + w_derived.w2) * f0)),
        b_11    (b_denom * (static_cast<floatT>(2.0) * u_derived.u * r_consts.r1_1 + (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * r_consts.r2_1 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u_derived.u2 + w_derived.w2) * f1)),
        b_12    (b_denom * (static_cast<floatT>(2.0) * u_derived.u * r_consts.r1_2 + (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * r_consts.r2_2 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u_derived.u2 + w_derived.w2) * f2)),
        b_20    (b_denom * (r_consts.r1_0 + static_cast<floatT>(3.0) * u_derived.u * r_consts.r2_0 - static_cast<floatT>(24.0) * u_derived.u * f0)),
        b_21    (b_denom * (r_consts.r1_1 + static_cast<floatT>(3.0) * u_derived.u * r_consts.r2_1 - static_cast<floatT>(24.0) * u_derived.u * f1)),
        b_22    (b_denom * (r_consts.r1_2 + static_cast<floatT>(3.0) * u_derived.u * r_consts.r2_2 - static_cast<floatT>(24.0) * u_derived.u * f2))
        {}
    };

    // Constants used during the calculation of the matrix exponential via Cayley-Hamilton
    struct ExpConstants
    {
        Matrix_3x3 Mat;
        Matrix_3x3 Mat2;
        // Both c0 and c1 (det(Mat_in) and 0.5*tr(Mat_in^2) respectively) are real since we consider Hermitian matrices
        floatT c0;
        floatT c1;
        floatT c0_max;
        // Theta, u, w are real parameters derived from c0 and c0_max
        floatT theta;
        // Auxiliary variables depending on u (including u itself)
        uDerivedConstants u_derived;
        // Auxiliary variables depending on w (including w itself)
        wDerivedConstants w_derived;
        // Denominator of coefficients f_0, f_1, f_2
        std::complex<floatT> denom;
        // h_0, h_1, h_2 functions used during calculation of f_1, f_2, f_3
        // TODO: Numerically problematic if w -> 3u -> sqrt(3)/2 as c0 -> -c0_max?
        // Can be circumvented by using symmetry relation of f_j, but is that really necessary here?
        // If so, we only want to check once if c0 is negative
        std::complex<floatT> h0;
        std::complex<floatT> h1;
        std::complex<floatT> h2;
        //-----
        ExpConstants(const Matrix_3x3& Mat_in) noexcept :
        Mat       (Mat_in),
        Mat2      (Mat_in * Mat_in),
        c0        (static_cast<floatT>(1.0/3.0) * std::real((Mat_in * Mat2).trace())),
        c1        (static_cast<floatT>(0.5) * std::real(Mat2.trace())),
        c0_max    (static_cast<floatT>(2.0) * std::pow(c1 / static_cast<floatT>(3.0), static_cast<floatT>(1.5))),
        theta     (std::acos(c0/c0_max)),
        u_derived (c1, theta),
        w_derived (c1, theta),
        denom     (static_cast<floatT>(1.0) / (static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2)),
        h0        ((u_derived.u2 - w_derived.w2) * u_derived.exp_2iu + u_derived.exp_miu * (static_cast<floatT>(8.0) * u_derived.u2 * w_derived.cosw + static_cast<floatT>(2.0) * u_derived.u * w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 + w_derived.w2))),
        h1        (static_cast<floatT>(2.0) * u_derived.u * u_derived.exp_2iu - u_derived.exp_miu * (static_cast<floatT>(2.0) * u_derived.u * w_derived.cosw - (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * w_derived.i_xi0)),
        h2        (u_derived.exp_2iu - u_derived.exp_miu * (w_derived.cosw + static_cast<floatT>(3.0) * u_derived.u * w_derived.i_xi0))
        {}
    };

    // Constants used during the calculation of the derivative of the matrix exponential via Cayley-Hamilton
    struct ExpDerivativeConstants
    {
        Matrix_3x3           Mat;
        Matrix_3x3           Mat2;
        // Both c0 and c1 (det(Mat_in) and 0.5*tr(Mat_in^2) respectively) are real since we consider Hermitian matrices
        floatT               c0;
        floatT               c1;
        floatT               c0_max;
        // Theta, u, w are real parameters derived from c0 and c0_max
        floatT               theta;
        // Auxiliary variables depending on u (including u itself)
        uDerivedConstants    u_derived;
        // Auxiliary variables depending on w (including w itself)
        wDerivedConstants    w_derived;
        std::complex<floatT> xi_0;
        std::complex<floatT> xi_1;
        // Denominator of coefficients f_1, f_2, f_3
        std::complex<floatT> denom;
        // h_0, h_1, h_2 functions used during calculation of f_1, f_2, f_3
        // TODO: Numerically problematic if w -> 3u -> sqrt(3)/2 as c0 -> -c0_max?
        // Can be circumvented by using symmetry relation of f_j, but is that really necessary here?
        // If so, we only want to check once if c0 is negative
        std::complex<floatT> h0;
        std::complex<floatT> h1;
        std::complex<floatT> h2;
        // f_0, f_1, f_2 functions (in contrast to ExpConstants, it is more convernient to explicitly have f_i here)
        std::complex<floatT> f0;
        std::complex<floatT> f1;
        std::complex<floatT> f2;
        // All ri_j coefficients
        rConstants           r_consts;
        // All b_ij coefficients
        bConstants           b_consts;
        // Coefficients B_i
        Matrix_3x3           B_1;
        Matrix_3x3           B_2;
        //-----
        ExpDerivativeConstants(const Matrix_3x3& Mat_in) noexcept :
        Mat       (Mat_in),
        Mat2      (Mat_in * Mat_in),
        c0        (static_cast<floatT>(1.0/3.0) * std::real((Mat_in * Mat2).trace())),
        c1        (static_cast<floatT>(0.5) * std::real(Mat2.trace())),
        c0_max    (static_cast<floatT>(2.0) * std::pow(c1 / static_cast<floatT>(3.0), static_cast<floatT>(1.5))),
        theta     (std::acos(c0/c0_max)),
        u_derived (c1, theta),
        w_derived (c1, theta),
        xi_0      (return_xi_0(w_derived.w)),
        xi_1      (return_xi_1(w_derived.w)),
        denom     (static_cast<floatT>(1.0) / (static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2)),
        h0        ((u_derived.u2 - w_derived.w2) * u_derived.exp_2iu + u_derived.exp_miu * (static_cast<floatT>(8.0) * u_derived.u2 * w_derived.cosw + static_cast<floatT>(2.0) * u_derived.u * w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 + w_derived.w2))),
        h1        (static_cast<floatT>(2.0) * u_derived.u * u_derived.exp_2iu - u_derived.exp_miu * (static_cast<floatT>(2.0) * u_derived.u * w_derived.cosw - w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2))),
        h2        (u_derived.exp_2iu - u_derived.exp_miu * (w_derived.cosw + static_cast<floatT>(3.0) * w_derived.i_xi0 * u_derived.u)),
        f0        (denom * h0),
        f1        (denom * h1),
        f2        (denom * h2),
        r_consts  (u_derived, w_derived, xi_0, xi_1),
        b_consts  (u_derived, w_derived, r_consts, f0, f1, f2),
        B_1       (b_consts.b_10 * Matrix_3x3::Identity() + b_consts.b_11 * Mat + b_consts.b_12 * Mat2),
        B_2       (b_consts.b_20 * Matrix_3x3::Identity() + b_consts.b_21 * Mat + b_consts.b_22 * Mat2)
        {}
    };

    // Computes exp(i * Mat), where Mat is a traceless hermitian 3x3 matrix
    [[nodiscard]]
    Matrix_SU3 exp(const Matrix_3x3& Mat) noexcept
    {
        ExpConstants exp_consts(Mat);
        return exp_consts.denom * (exp_consts.h0 * Matrix_3x3::Identity() + exp_consts.h1 * exp_consts.Mat + exp_consts.h2 * exp_consts.Mat2);
    }
    // Version where we reuse already computed ExpConstants
    [[nodiscard]]
    Matrix_SU3 exp(const ExpConstants& exp_consts) noexcept
    {
        return exp_consts.denom * (exp_consts.h0 * Matrix_3x3::Identity() + exp_consts.h1 * exp_consts.Mat + exp_consts.h2 * exp_consts.Mat2);
    }
    // Version where we reuse already computed ExpDerivativeConstants
    [[nodiscard]]
    Matrix_SU3 exp(const ExpDerivativeConstantsConstants& expd_consts) noexcept
    {
        return expd_consts.denom * (expd_consts.h0 * Matrix_3x3::Identity() + expd_consts.h1 * expd_consts.Mat + expd_consts.h2 * expd_consts.Mat2);
    }
}

#endif // LETTUCE_SU3_EXP_HPP
