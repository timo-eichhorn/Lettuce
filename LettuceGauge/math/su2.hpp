#ifndef LETTUCE_SU2_HPP
#define LETTUCE_SU2_HPP

// Non-standard library headers
#include "../defines.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <complex>
//----------------------------------------
// Standard C headers
// ...

//----------------------------------------
// Provides SU(2) subgroup class and functions to extract SU(2) matrices from SU(3) matrices/embed SU(2) matrices in SU(3) matrices
// Note that this is class is also used for representing the sum of SU(2) matrices, which is still proportional to a SU(2) element, but generally not a SU(2) element itself
// Used during heat bath and overrelaxation update

template<typename floatT>
class SU2_comp
{
    public:
        std::complex<floatT> e11, e12;

        // SU2_comp() noexcept {};
        SU2_comp() noexcept = default;

        SU2_comp(std::complex<floatT> e11_in, std::complex<floatT> e12_in) noexcept : e11(e11_in), e12(e12_in) {}

        friend SU2_comp operator+(const SU2_comp& Mat1, const SU2_comp& Mat2) noexcept
        {
            return {Mat1.e11 + Mat2.e11, Mat1.e12 + Mat2.e12};
        }
        friend SU2_comp operator-(const SU2_comp& Mat1, const SU2_comp& Mat2) noexcept
        {
            return {Mat1.e11 - Mat2.e11, Mat1.e12 - Mat2.e12};
        }
        friend SU2_comp operator*(const SU2_comp& Mat1, const SU2_comp& Mat2) noexcept
        {
            return {Mat1.e11 * Mat2.e11 - Mat1.e12 * std::conj(Mat2.e12), Mat1.e11 * Mat2.e12 + Mat1.e12 * std::conj(Mat2.e11)};
        }

        // TODO: Is this necessary or is it sufficient to define the scalar multiplication for complex numbers?
        friend SU2_comp operator*(const floatT a, const SU2_comp& Mat) noexcept
        {
            return {a * Mat.e11, a * Mat.e12};
        }
        friend SU2_comp operator*(const SU2_comp& Mat, const floatT a) noexcept
        {
            return {Mat.e11 * a, Mat.e12 * a};
        }
        friend SU2_comp operator*(const std::complex<floatT> a, const SU2_comp& Mat) noexcept
        {
            return {a * Mat.e11, a * Mat.e12};
        }
        friend SU2_comp operator*(const SU2_comp& Mat, const std::complex<floatT> a) noexcept
        {
            return {Mat.e11 * a, Mat.e12 * a};
        }
        // TODO: Implement scalar division operator / and operator /=?

        SU2_comp &operator=(const SU2_comp& Mat) noexcept
        {
            e11 = Mat.e11;
            e12 = Mat.e12;
            return *this;
        }
        SU2_comp &operator+=(const SU2_comp& Mat) noexcept
        {
            e11 += Mat.e11;
            e12 += Mat.e12;
            return *this;
        }
        SU2_comp &operator-=(const SU2_comp& Mat) noexcept
        {
            e11 -= Mat.e11;
            e12 -= Mat.e12;
            return *this;
        }
        SU2_comp &operator*=(const SU2_comp& Mat) noexcept
        {
            *this = *this * Mat;
            return *this;
        }
        SU2_comp &operator*=(const floatT a) noexcept
        {
            *this = *this * a;
            return *this;
        }
        SU2_comp &operator*=(const std::complex<floatT> a) noexcept
        {
            *this = *this * a;
            return *this;
        }

        SU2_comp adjoint() const noexcept
        {
            return {std::conj(e11), -e12};
        }

        // The determinant of a SU(2) matrix is 1, but since this class is also used to represent matrices propotional to SU(2) elements this function doesn't always return 1
        floatT det() const noexcept
        {
            return std::real(e11) * std::real(e11) + std::imag(e11) * std::imag(e11) + std::real(e12) * std::real(e12) + std::imag(e12) * std::imag(e12);
        }

        // The determinant of a SU(2) matrix is 1, but since this class is also used to represent matrices propotional to SU(2) elements this function doesn't always return 1
        floatT det_sqrt() const noexcept
        {
            return std::sqrt(det());
        }
};

// Directly embed SU(2) matrix into SU(3) matrix

template<typename floatT>
Matrix_3x3 Embed01(const SU2_comp<floatT>& Mat) noexcept
{
    Matrix_3x3 Mat_embedded;
    Mat_embedded <<             Mat.e11,            Mat.e12, 0,
                    -std::conj(Mat.e12), std::conj(Mat.e11), 0,
                                      0,                  0, 1;
    return Mat_embedded;
}

template<typename floatT>
Matrix_3x3 Embed02(const SU2_comp<floatT>& Mat) noexcept
{
    Matrix_3x3 Mat_embedded;
    Mat_embedded <<             Mat.e11, 0,            Mat.e12,
                                      0, 1,                  0,
                    -std::conj(Mat.e12), 0, std::conj(Mat.e11);
    return Mat_embedded;
}

template<typename floatT>
Matrix_3x3 Embed12(const SU2_comp<floatT>& Mat) noexcept
{
    Matrix_3x3 Mat_embedded;
    Mat_embedded << 1,                   0,                  0,
                    0,             Mat.e11,            Mat.e12,
                    0, -std::conj(Mat.e12), std::conj(Mat.e11);
    return Mat_embedded;
}

// Extract SU(2) matrix from SU(3) matrix via projection (we obviously can't directly extract subblocks since that wouldn't generally be in SU(2))
// TODO: Write function that directly extracts the product? Might lead to better performance since we need fewer multiplications

template<typename floatT>
SU2_comp<floatT> Extract01(const Matrix_3x3& Mat) noexcept
{
    std::complex<floatT> tmp1 {static_cast<floatT>(0.5) * (Mat(0, 0) + std::conj(Mat(1, 1)))};
    std::complex<floatT> tmp2 {static_cast<floatT>(0.5) * (Mat(0, 1) - std::conj(Mat(1, 0)))};
    return {tmp1, tmp2};
}

// TODO: For symmetry reasons, make this (2, 0) instead of (0, 2)?
template<typename floatT>
SU2_comp<floatT> Extract02(const Matrix_3x3& Mat) noexcept
{
    std::complex<floatT> tmp1 {static_cast<floatT>(0.5) * (Mat(0, 0) + std::conj(Mat(2, 2)))};
    std::complex<floatT> tmp2 {static_cast<floatT>(0.5) * (Mat(0, 2) - std::conj(Mat(2, 0)))};
    return {tmp1, tmp2};
}

template<typename floatT>
SU2_comp<floatT> Extract12(const Matrix_3x3& Mat) noexcept
{
    std::complex<floatT> tmp1 {static_cast<floatT>(0.5) * (Mat(1, 1) + std::conj(Mat(2, 2)))};
    std::complex<floatT> tmp2 {static_cast<floatT>(0.5) * (Mat(1, 2) - std::conj(Mat(2, 1)))};
    return {tmp1, tmp2};
}

// TODO: Test if this is faster. Preliminary test pretty much show no difference, probably due to Eigen's expression templates

// SU2_comp Extract01_new(const Matrix_3x3& Mat1, const Matrix_3x3 Mat2)
// {
//     std::complex<floatT> prod_00 {Mat1(0, 0) * Mat2(0, 0) + Mat1(0, 1) * Mat2(1, 0) + Mat1(0, 2) * Mat2(2, 0)};
//     std::complex<floatT> prod_01 {Mat1(0, 0) * Mat2(0, 1) + Mat1(0, 1) * Mat2(1, 1) + Mat1(0, 2) * Mat2(2, 1)};
//     std::complex<floatT> prod_10 {Mat1(1, 0) * Mat2(0, 0) + Mat1(1, 1) * Mat2(1, 0) + Mat1(1, 2) * Mat2(2, 0)};
//     std::complex<floatT> prod_11 {Mat1(1, 0) * Mat2(0, 1) + Mat1(1, 1) * Mat2(1, 1) + Mat1(1, 2) * Mat2(2, 1)};
//     std::complex<floatT> tmp1   {static_cast<floatT>(0.5) * (prod_00 + std::conj(prod_11))};
//     std::complex<floatT> tmp2   {static_cast<floatT>(0.5) * (prod_01 - std::conj(prod_10))};
//     return {tmp1, tmp2};
// }

// SU2_comp Extract02_new(const Matrix_3x3& Mat1, const Matrix_3x3 Mat2)
// {
//     std::complex<floatT> prod_00 {Mat1(0, 0) * Mat2(0, 0) + Mat1(0, 1) * Mat2(1, 0) + Mat1(0, 2) * Mat2(2, 0)};
//     std::complex<floatT> prod_02 {Mat1(0, 0) * Mat2(0, 2) + Mat1(0, 1) * Mat2(1, 2) + Mat1(0, 2) * Mat2(2, 2)};
//     std::complex<floatT> prod_20 {Mat1(2, 0) * Mat2(0, 0) + Mat1(2, 1) * Mat2(1, 0) + Mat1(2, 2) * Mat2(2, 0)};
//     std::complex<floatT> prod_22 {Mat1(2, 0) * Mat2(0, 2) + Mat1(2, 1) * Mat2(1, 2) + Mat1(2, 2) * Mat2(2, 2)};
//     std::complex<floatT> tmp1   {static_cast<floatT>(0.5) * (prod_00 + std::conj(prod_22))};
//     std::complex<floatT> tmp2   {static_cast<floatT>(0.5) * (prod_02 - std::conj(prod_20))};
//     return {tmp1, tmp2};
// }

// SU2_comp Extract12_new(const Matrix_3x3& Mat1, const Matrix_3x3 Mat2)
// {
//     std::complex<floatT> prod_11 {Mat1(1, 0) * Mat2(0, 1) + Mat1(1, 1) * Mat2(1, 1) + Mat1(1, 2) * Mat2(2, 1)};
//     std::complex<floatT> prod_12 {Mat1(1, 0) * Mat2(0, 2) + Mat1(1, 1) * Mat2(1, 2) + Mat1(1, 2) * Mat2(2, 2)};
//     std::complex<floatT> prod_21 {Mat1(2, 0) * Mat2(0, 1) + Mat1(2, 1) * Mat2(1, 1) + Mat1(2, 2) * Mat2(2, 1)};
//     std::complex<floatT> prod_22 {Mat1(2, 0) * Mat2(0, 2) + Mat1(2, 1) * Mat2(1, 2) + Mat1(2, 2) * Mat2(2, 2)};
//     std::complex<floatT> tmp1   {static_cast<floatT>(0.5) * (prod_11 + std::conj(prod_22))};
//     std::complex<floatT> tmp2   {static_cast<floatT>(0.5) * (prod_12 - std::conj(prod_21))};
//     return {tmp1, tmp2};
// }

#endif // LETTUCE_SU2_HPP
