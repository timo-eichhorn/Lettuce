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
#include <limits>
//----------------------------------------
// Standard C headers
#include <cmath>

namespace SU3
{
    [[nodiscard]]
    floatT return_xi_0(const floatT w) noexcept
    {
        // TODO: It seems like we get the exact same results using this and std::sin(w)/w for extremely small non-zero numbers, so perhaps this is unneccessary
        //       Maybe it is faster than std::sin(), but benchmark first!
        //       In the range slighltly below 0.05, the series and std::sin(w)/w mostly match
        if (std::abs(w) <= static_cast<floatT>(0.05))
        {
            // 6th order Taylor series of sin(w)/w
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
        // Funnily this seems much more instable than xi_0
        // Threshold is not tuned exactly, but for 0.12 the series expansion seems to be slightly less accurate, while for 0.11 it seems more accurate
        // Somewhere around 0.115 seems to be a decent choice
        if (std::abs(w) <= static_cast<floatT>(0.115))
        {
            // 6th order Taylor series of cos(w)/w^2 - sin(w)/w^3
            floatT w2 {w * w};
            return -static_cast<floatT>(1.0/3.0) + static_cast<floatT>(1.0/30.0) * w2 * (static_cast<floatT>(1.0) - static_cast<floatT>(1.0/28.0) * w2 * (static_cast<floatT>(1.0) - static_cast<floatT>(1.0/54.0) * w2));
        }
        else
        {
            return std::cos(w)/(w * w) - std::sin(w)/(w * w * w);
        }
    }

    struct uDerivedConstants
    {
        floatT               u;
        floatT               u2;
        std::complex<floatT> exp_miu;
        std::complex<floatT> exp_2iu;
        //-----
        // We need a default constructor to be able to create a lattice holding these constants
        uDerivedConstants() = default;
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
        //-----
        // We need a default constructor to be able to create a lattice holding these constants
        wDerivedConstants() = default;
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
        // We need a default constructor to be able to create a lattice holding these constants
        rConstants() = default;
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
        floatT               b_denom;
        std::complex<floatT> b_10;
        std::complex<floatT> b_11;
        std::complex<floatT> b_12;
        std::complex<floatT> b_20;
        std::complex<floatT> b_21;
        std::complex<floatT> b_22;
        //-----
        // We need a default constructor to be able to create a lattice holding these constants
        bConstants() = default;
        // TODO: Encapsulate f_i (and h_i?) in a struct?
        // bConstants(const uDerivedConstants& u_derived, const wDerivedConstants& w_derived, const rConstants& r_consts, const std::complex<floatT> f0, const std::complex<floatT> f1, const std::complex<floatT> f2) noexcept :
        // b_denom (static_cast<floatT>(0.5) / (std::pow(static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2, 2))),
        // b_10    (b_denom * (static_cast<floatT>(2.0) * u_derived.u * r_consts.r1_0 + (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * r_consts.r2_0 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u_derived.u2 + w_derived.w2) * f0)),
        // b_11    (b_denom * (static_cast<floatT>(2.0) * u_derived.u * r_consts.r1_1 + (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * r_consts.r2_1 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u_derived.u2 + w_derived.w2) * f1)),
        // b_12    (b_denom * (static_cast<floatT>(2.0) * u_derived.u * r_consts.r1_2 + (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * r_consts.r2_2 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u_derived.u2 + w_derived.w2) * f2)),
        // b_20    (b_denom * (r_consts.r1_0 - static_cast<floatT>(3.0) * u_derived.u * r_consts.r2_0 - static_cast<floatT>(24.0) * u_derived.u * f0)),
        // b_21    (b_denom * (r_consts.r1_1 - static_cast<floatT>(3.0) * u_derived.u * r_consts.r2_1 - static_cast<floatT>(24.0) * u_derived.u * f1)),
        // b_22    (b_denom * (r_consts.r1_2 - static_cast<floatT>(3.0) * u_derived.u * r_consts.r2_2 - static_cast<floatT>(24.0) * u_derived.u * f2))
        // {}
        bConstants(const uDerivedConstants& u_derived, const wDerivedConstants& w_derived, const rConstants& r_consts, const std::complex<floatT> f0, const std::complex<floatT> f1, const std::complex<floatT> f2, const bool signflip) noexcept
        {
            // TODO: Check if b_denom is NaN by checking if 9.0 * u^2 - w^2 is too small
            if (std::pow(static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2, 2) < std::numeric_limits<floatT>::min())
            {
                b_denom = static_cast<floatT>(0.0);
            }
            else
            {
                b_denom = static_cast<floatT>(0.5) / (std::pow(static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2, 2));
            }
            // If c0 -> -c0_max we need to use a symmetry relation to avoid numerical instabilites
            if (signflip)
            {
                // b_denom =  static_cast<floatT>(0.5) / (std::pow(static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2, 2));
                b_10    =  b_denom * std::conj(static_cast<floatT>(2.0) * u_derived.u * r_consts.r1_0 + (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * r_consts.r2_0 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u_derived.u2 + w_derived.w2) * f0);
                b_11    = -b_denom * std::conj(static_cast<floatT>(2.0) * u_derived.u * r_consts.r1_1 + (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * r_consts.r2_1 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u_derived.u2 + w_derived.w2) * f1);
                b_12    =  b_denom * std::conj(static_cast<floatT>(2.0) * u_derived.u * r_consts.r1_2 + (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * r_consts.r2_2 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u_derived.u2 + w_derived.w2) * f2);
                b_20    = -b_denom * std::conj(r_consts.r1_0 - static_cast<floatT>(3.0) * u_derived.u * r_consts.r2_0 - static_cast<floatT>(24.0) * u_derived.u * f0);
                b_21    =  b_denom * std::conj(r_consts.r1_1 - static_cast<floatT>(3.0) * u_derived.u * r_consts.r2_1 - static_cast<floatT>(24.0) * u_derived.u * f1);
                b_22    = -b_denom * std::conj(r_consts.r1_2 - static_cast<floatT>(3.0) * u_derived.u * r_consts.r2_2 - static_cast<floatT>(24.0) * u_derived.u * f2);
            }
            else
            {
                // b_denom = static_cast<floatT>(0.5) / (std::pow(static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2, 2));
                b_10    = b_denom * (static_cast<floatT>(2.0) * u_derived.u * r_consts.r1_0 + (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * r_consts.r2_0 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u_derived.u2 + w_derived.w2) * f0);
                b_11    = b_denom * (static_cast<floatT>(2.0) * u_derived.u * r_consts.r1_1 + (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * r_consts.r2_1 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u_derived.u2 + w_derived.w2) * f1);
                b_12    = b_denom * (static_cast<floatT>(2.0) * u_derived.u * r_consts.r1_2 + (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * r_consts.r2_2 - static_cast<floatT>(2.0) * (static_cast<floatT>(15.0) * u_derived.u2 + w_derived.w2) * f2);
                b_20    = b_denom * (r_consts.r1_0 - static_cast<floatT>(3.0) * u_derived.u * r_consts.r2_0 - static_cast<floatT>(24.0) * u_derived.u * f0);
                b_21    = b_denom * (r_consts.r1_1 - static_cast<floatT>(3.0) * u_derived.u * r_consts.r2_1 - static_cast<floatT>(24.0) * u_derived.u * f1);
                b_22    = b_denom * (r_consts.r1_2 - static_cast<floatT>(3.0) * u_derived.u * r_consts.r2_2 - static_cast<floatT>(24.0) * u_derived.u * f2);
            }
        }
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
        // Flip sign of c0 for better numerical stabilty (we always work with positive c0)
        bool   signflip;
        // Theta, u, w are real parameters derived from c0 and c0_max
        floatT theta;
        // Auxiliary variables depending on u (including u itself)
        uDerivedConstants u_derived;
        // Auxiliary variables depending on w (including w itself)
        wDerivedConstants w_derived;
        // Denominator of coefficients f_0, f_1, f_2
        floatT denom;
        // h_0, h_1, h_2 functions used during calculation of f_1, f_2, f_3
        std::complex<floatT> h0;
        std::complex<floatT> h1;
        std::complex<floatT> h2;
        //-----
        // We need a default constructor to be able to create a lattice holding these constants
        ExpConstants() = default;
        ExpConstants(const Matrix_3x3& Mat_in) noexcept :
        Mat       (Mat_in),
        Mat2      (Mat_in * Mat_in),
        // We want to avoid negative c0 due to numerical instabilities, so always use the absolute value and remember the sign in signflip
        c0        (std::abs(static_cast<floatT>(1.0/3.0) * std::real((Mat_in * Mat2).trace()))),
        c1        (static_cast<floatT>(0.5) * std::real(Mat2.trace())),
        c0_max    (static_cast<floatT>(2.0) * std::pow(c1 / static_cast<floatT>(3.0), static_cast<floatT>(1.5))),
        signflip  (static_cast<floatT>(1.0/3.0) * std::real((Mat_in * Mat2).trace()) < static_cast<floatT>(0.0)),
        // TODO: On paper c0/c0_max <= 1, but if c0 = c0_max = 0 the division returns -NaN. Need to handle this using fmin, which trats NaNs as missing data
        theta     (std::acos(std::fmin(c0/c0_max, static_cast<floatT>(1.0)))),
        u_derived (c1, theta),
        w_derived (c1, theta)
        // denom     (static_cast<floatT>(1.0) / (static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2)),
        // h0        ((u_derived.u2 - w_derived.w2) * u_derived.exp_2iu + u_derived.exp_miu * (static_cast<floatT>(8.0) * u_derived.u2 * w_derived.cosw + static_cast<floatT>(2.0) * u_derived.u * w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 + w_derived.w2))),
        // h1        (static_cast<floatT>(2.0) * u_derived.u * u_derived.exp_2iu - u_derived.exp_miu * (static_cast<floatT>(2.0) * u_derived.u * w_derived.cosw - (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * w_derived.i_xi0)),
        // h2        (u_derived.exp_2iu - u_derived.exp_miu * (w_derived.cosw + static_cast<floatT>(3.0) * u_derived.u * w_derived.i_xi0))
        {
            if (signflip)
            {
                denom =  static_cast<floatT>(1.0) / (static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2);
                h0    =  std::conj((u_derived.u2 - w_derived.w2) * u_derived.exp_2iu + u_derived.exp_miu * (static_cast<floatT>(8.0) * u_derived.u2 * w_derived.cosw + static_cast<floatT>(2.0) * u_derived.u * w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 + w_derived.w2)));
                h1    = -std::conj(static_cast<floatT>(2.0) * u_derived.u * u_derived.exp_2iu - u_derived.exp_miu * (static_cast<floatT>(2.0) * u_derived.u * w_derived.cosw - w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2)));
                h2    =  std::conj(u_derived.exp_2iu - u_derived.exp_miu * (w_derived.cosw + static_cast<floatT>(3.0) * u_derived.u * w_derived.i_xi0));
            }
            else
            {
                denom = (static_cast<floatT>(1.0) / (static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2));
                h0    = ((u_derived.u2 - w_derived.w2) * u_derived.exp_2iu + u_derived.exp_miu * (static_cast<floatT>(8.0) * u_derived.u2 * w_derived.cosw + static_cast<floatT>(2.0) * u_derived.u * w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 + w_derived.w2)));
                h1    = (static_cast<floatT>(2.0) * u_derived.u * u_derived.exp_2iu - u_derived.exp_miu * (static_cast<floatT>(2.0) * u_derived.u * w_derived.cosw - w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2)));
                h2    = (u_derived.exp_2iu - u_derived.exp_miu * (w_derived.cosw + static_cast<floatT>(3.0) * u_derived.u * w_derived.i_xi0));
            }
            // TODO: denom can still be 0
            // TODO: If we check here, do we still need to check during assignment of theta above? Can anything go wrong if we pass incorrect values to u_derived and w_derived?
            if (c0_max < std::numeric_limits<floatT>::min())
            {
                denom = static_cast<floatT>(1.0);
                h0    = static_cast<floatT>(1.0);
                h1    = static_cast<floatT>(0.0);
                h2    = static_cast<floatT>(0.0);
            }
        }
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
        // Flip sign of c0 for better numerical stabilty (we always work with positive c0)
        bool                 signflip;
        // Theta, u, w are real parameters derived from c0 and c0_max
        floatT               theta;
        // Auxiliary variables depending on u (including u itself)
        uDerivedConstants    u_derived;
        // Auxiliary variables depending on w (including w itself)
        wDerivedConstants    w_derived;
        std::complex<floatT> xi_0;
        std::complex<floatT> xi_1;
        // Denominator of coefficients f_1, f_2, f_3
        floatT denom;
        // h_0, h_1, h_2 functions used during calculation of f_1, f_2, f_3
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
        // We need a default constructor to be able to create a lattice holding these constants
        ExpDerivativeConstants() = default;
        ExpDerivativeConstants(const Matrix_3x3& Mat_in) noexcept :
        Mat       (Mat_in),
        Mat2      (Mat_in * Mat_in),
        // We want to avoid negative c0 due to numerical instabilities, so always use the absolute value and remember the sign in signflip
        c0        (std::abs(static_cast<floatT>(1.0/3.0) * std::real((Mat_in * Mat2).trace()))),
        c1        (static_cast<floatT>(0.5) * std::real(Mat2.trace())),
        c0_max    (static_cast<floatT>(2.0) * std::pow(c1 / static_cast<floatT>(3.0), static_cast<floatT>(1.5))),
        signflip  (static_cast<floatT>(1.0/3.0) * std::real((Mat_in * Mat2).trace()) < static_cast<floatT>(0.0)),
        // TODO: On paper c0/c0_max <= 1, but if c0 = c0_max = 0 the division returns -NaN. Need to handle this using fmin, which trats NaNs as missing data
        //       Also add this to ExpDerivativeConstants below
        theta     (std::acos(std::fmin(c0/c0_max, static_cast<floatT>(1.0)))),
        u_derived (c1, theta),
        w_derived (c1, theta),
        xi_0      (return_xi_0(w_derived.w)),
        xi_1      (return_xi_1(w_derived.w))
        // denom     (static_cast<floatT>(1.0) / (static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2)),
        // h0        ((u_derived.u2 - w_derived.w2) * u_derived.exp_2iu + u_derived.exp_miu * (static_cast<floatT>(8.0) * u_derived.u2 * w_derived.cosw + static_cast<floatT>(2.0) * u_derived.u * w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 + w_derived.w2))),
        // h1        (static_cast<floatT>(2.0) * u_derived.u * u_derived.exp_2iu - u_derived.exp_miu * (static_cast<floatT>(2.0) * u_derived.u * w_derived.cosw - (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2) * w_derived.i_xi0)),
        // h2        (u_derived.exp_2iu - u_derived.exp_miu * (w_derived.cosw + static_cast<floatT>(3.0) * u_derived.u * w_derived.i_xi0)),
        // f0        (denom * h0),
        // f1        (denom * h1),
        // f2        (denom * h2),
        // r_consts  (u_derived, w_derived, xi_0, xi_1),
        // b_consts  (u_derived, w_derived, r_consts, f0, f1, f2, signflip),
        // B_1       (b_consts.b_10 * Matrix_3x3::Identity() + b_consts.b_11 * Mat + b_consts.b_12 * Mat2),
        // B_2       (b_consts.b_20 * Matrix_3x3::Identity() + b_consts.b_21 * Mat + b_consts.b_22 * Mat2)
        {
            if (signflip)
            {
                denom =  static_cast<floatT>(1.0) / (static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2);
                h0    =  std::conj((u_derived.u2 - w_derived.w2) * u_derived.exp_2iu + u_derived.exp_miu * (static_cast<floatT>(8.0) * u_derived.u2 * w_derived.cosw + static_cast<floatT>(2.0) * u_derived.u * w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 + w_derived.w2)));
                h1    = -std::conj(static_cast<floatT>(2.0) * u_derived.u * u_derived.exp_2iu - u_derived.exp_miu * (static_cast<floatT>(2.0) * u_derived.u * w_derived.cosw - w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2)));
                h2    =  std::conj(u_derived.exp_2iu - u_derived.exp_miu * (w_derived.cosw + static_cast<floatT>(3.0) * u_derived.u * w_derived.i_xi0));
            }
            else
            {
                denom = (static_cast<floatT>(1.0) / (static_cast<floatT>(9.0) * u_derived.u2 - w_derived.w2));
                h0    = ((u_derived.u2 - w_derived.w2) * u_derived.exp_2iu + u_derived.exp_miu * (static_cast<floatT>(8.0) * u_derived.u2 * w_derived.cosw + static_cast<floatT>(2.0) * u_derived.u * w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 + w_derived.w2)));
                h1    = (static_cast<floatT>(2.0) * u_derived.u * u_derived.exp_2iu - u_derived.exp_miu * (static_cast<floatT>(2.0) * u_derived.u * w_derived.cosw - w_derived.i_xi0 * (static_cast<floatT>(3.0) * u_derived.u2 - w_derived.w2)));
                h2    = (u_derived.exp_2iu - u_derived.exp_miu * (w_derived.cosw + static_cast<floatT>(3.0) * u_derived.u * w_derived.i_xi0));
            }
            // TODO: denom can still be 0
            // TODO: If we check here, do we still need to check during assignment of theta above? Can anything go wrong if we pass incorrect values to u_derived and w_derived?
            if (c0_max < std::numeric_limits<floatT>::min())
            {
                denom = static_cast<floatT>(1.0);
                h0    = static_cast<floatT>(1.0);
                h1    = static_cast<floatT>(0.0);
                h2    = static_cast<floatT>(0.0);
            }
            f0       = denom * h0;
            f1       = denom * h1;
            f2       = denom * h2;
            r_consts = rConstants(u_derived, w_derived, xi_0, xi_1);
            b_consts = bConstants(u_derived, w_derived, r_consts, f0, f1, f2, signflip);
            B_1      = b_consts.b_10 * Matrix_3x3::Identity() + b_consts.b_11 * Mat + b_consts.b_12 * Mat2;
            B_2      = b_consts.b_20 * Matrix_3x3::Identity() + b_consts.b_21 * Mat + b_consts.b_22 * Mat2;
        }
        ExpDerivativeConstants(const ExpConstants& ExpConstants_in) noexcept :
        Mat       (ExpConstants_in.Mat),
        Mat2      (ExpConstants_in.Mat2),
        c0        (ExpConstants_in.c0),
        c1        (ExpConstants_in.c1),
        c0_max    (ExpConstants_in.c0_max),
        signflip  (ExpConstants_in.signflip),
        theta     (ExpConstants_in.theta),
        u_derived (ExpConstants_in.u_derived),
        w_derived (ExpConstants_in.w_derived),
        xi_0      (return_xi_0(w_derived.w)),
        xi_1      (return_xi_1(w_derived.w)),
        denom     (ExpConstants_in.denom),
        h0        (ExpConstants_in.h0),
        h1        (ExpConstants_in.h1),
        h2        (ExpConstants_in.h2),
        f0        (denom * h0),
        f1        (denom * h1),
        f2        (denom * h2),
        r_consts  (u_derived, w_derived, xi_0, xi_1),
        b_consts  (u_derived, w_derived, r_consts, f0, f1, f2, signflip),
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
    Matrix_SU3 exp(const ExpDerivativeConstants& expd_consts) noexcept
    {
        return expd_consts.denom * (expd_consts.h0 * Matrix_3x3::Identity() + expd_consts.h1 * expd_consts.Mat + expd_consts.h2 * expd_consts.Mat2);
    }
} // namespace SU3

#endif // LETTUCE_SU3_EXP_HPP
