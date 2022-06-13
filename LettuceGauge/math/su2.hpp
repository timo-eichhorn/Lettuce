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
// Used during Heatbath and Overrelaxation update

template<typename floatT>
class SU2_comp
{
    public:
        std::complex<floatT> e11, e12;

        SU2_comp() noexcept {};

        SU2_comp(std::complex<floatT> e11_in, std::complex<floatT> e12_in) noexcept : e11(e11_in), e12(e12_in) {}

        friend SU2_comp operator+(const SU2_comp& mat1, const SU2_comp& mat2) noexcept
        {
            return {mat1.e11 + mat2.e11, mat1.e12 + mat2.e12};
        }
        friend SU2_comp operator-(const SU2_comp& mat1, const SU2_comp& mat2) noexcept
        {
            return {mat1.e11 - mat2.e11, mat1.e12 - mat2.e12};
        }
        friend SU2_comp operator*(const SU2_comp& mat1, const SU2_comp& mat2) noexcept
        {
            return {mat1.e11 * mat2.e11 - mat1.e12 * std::conj(mat2.e12), mat1.e11 * mat2.e12 + mat1.e12 * std::conj(mat2.e11)};
        }

        // TODO: Is this necessary or is it sufficient to define the scalar multiplication for complex numbers?
        friend SU2_comp operator*(const floatT a, const SU2_comp& mat) noexcept
        {
            return {a * mat.e11, a * mat.e12};
        }
        friend SU2_comp operator*(const SU2_comp& mat, const floatT a) noexcept
        {
            return {mat.e11 * a, mat.e12 * a};
        }
        friend SU2_comp operator*(const std::complex<floatT> a, const SU2_comp& mat) noexcept
        {
            return {a * mat.e11, a * mat.e12};
        }
        friend SU2_comp operator*(const SU2_comp& mat, const std::complex<floatT> a) noexcept
        {
            return {mat.e11 * a, mat.e12 * a};
        }
        // TODO: Implement scalar division operator / and operator /=?

        SU2_comp &operator=(const SU2_comp& mat) noexcept
        {
            e11 = mat.e11;
            e12 = mat.e12;
            return *this;
        }
        SU2_comp &operator+=(const SU2_comp& mat) noexcept
        {
            e11 += mat.e11;
            e12 += mat.e12;
            return *this;
        }
        SU2_comp &operator-=(const SU2_comp& mat) noexcept
        {
            e11 -= mat.e11;
            e12 -= mat.e12;
            return *this;
        }
        SU2_comp &operator*=(const SU2_comp& mat) noexcept
        {
            *this = *this * mat;
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

        floatT det_sq() const noexcept
        {
            return std::real(e11) * std::real(e11) + std::imag(e11) * std::imag(e11) + std::real(e12) * std::real(e12) + std::imag(e12) * std::imag(e12);
        }
};

// Directly embed SU(2) matrix into SU(3) matrix

template<typename floatT>
Matrix_3x3 Embed01(const SU2_comp<floatT>& mat) noexcept
{
    Matrix_3x3 mat_embedded;
    mat_embedded <<             mat.e11,            mat.e12, 0,
                    -std::conj(mat.e12), std::conj(mat.e11), 0,
                                      0,                  0, 1;
    return mat_embedded;
}

template<typename floatT>
Matrix_3x3 Embed02(const SU2_comp<floatT>& mat) noexcept
{
    Matrix_3x3 mat_embedded;
    mat_embedded <<             mat.e11, 0,            mat.e12,
                                      0, 1,                  0,
                    -std::conj(mat.e12), 0, std::conj(mat.e11);
    return mat_embedded;
}

template<typename floatT>
Matrix_3x3 Embed12(const SU2_comp<floatT>& mat) noexcept
{
    Matrix_3x3 mat_embedded;
    mat_embedded << 1,                   0,                  0,
                    0,             mat.e11,            mat.e12,
                    0, -std::conj(mat.e12), std::conj(mat.e11);
    return mat_embedded;
}

// Extract SU(2) matrix from SU(3) matrix via projection (we obviously can't directly extract subblocks since that wouldn't generally be in SU(2))
// TODO: Write function that directly extracts the product? Might lead to better performance since we need fewer multiplications

template<typename floatT>
SU2_comp<floatT> Extract01(const Matrix_3x3& mat) noexcept
{
    std::complex<floatT> temp1 {static_cast<floatT>(0.5) * (mat(0, 0) + std::conj(mat(1, 1)))};
    std::complex<floatT> temp2 {static_cast<floatT>(0.5) * (mat(0, 1) - std::conj(mat(1, 0)))};
    return {temp1, temp2};
}

// TODO: For symmetry reasons, make this (2, 0) instead of (0, 2)?
template<typename floatT>
SU2_comp<floatT> Extract02(const Matrix_3x3& mat) noexcept
{
    std::complex<floatT> temp1 {static_cast<floatT>(0.5) * (mat(0, 0) + std::conj(mat(2, 2)))};
    std::complex<floatT> temp2 {static_cast<floatT>(0.5) * (mat(0, 2) - std::conj(mat(2, 0)))};
    return {temp1, temp2};
}

template<typename floatT>
SU2_comp<floatT> Extract12(const Matrix_3x3& mat) noexcept
{
    std::complex<floatT> temp1 {static_cast<floatT>(0.5) * (mat(1, 1) + std::conj(mat(2, 2)))};
    std::complex<floatT> temp2 {static_cast<floatT>(0.5) * (mat(1, 2) - std::conj(mat(2, 1)))};
    return {temp1, temp2};
}

// TODO: Test if this is faster. Preliminary test pretty much show no difference, probably due to Eigen's expression templates

// SU2_comp Extract01_new(const Matrix_3x3& mat1, const Matrix_3x3 mat2)
// {
//     std::complex<floatT> prod_00 {mat1(0, 0) * mat2(0, 0) + mat1(0, 1) * mat2(1, 0) + mat1(0, 2) * mat2(2, 0)};
//     std::complex<floatT> prod_01 {mat1(0, 0) * mat2(0, 1) + mat1(0, 1) * mat2(1, 1) + mat1(0, 2) * mat2(2, 1)};
//     std::complex<floatT> prod_10 {mat1(1, 0) * mat2(0, 0) + mat1(1, 1) * mat2(1, 0) + mat1(1, 2) * mat2(2, 0)};
//     std::complex<floatT> prod_11 {mat1(1, 0) * mat2(0, 1) + mat1(1, 1) * mat2(1, 1) + mat1(1, 2) * mat2(2, 1)};
//     std::complex<floatT> temp1   {static_cast<floatT>(0.5) * (prod_00 + std::conj(prod_11))};
//     std::complex<floatT> temp2   {static_cast<floatT>(0.5) * (prod_01 - std::conj(prod_10))};
//     return {temp1, temp2};
// }

// SU2_comp Extract02_new(const Matrix_3x3& mat1, const Matrix_3x3 mat2)
// {
//     std::complex<floatT> prod_00 {mat1(0, 0) * mat2(0, 0) + mat1(0, 1) * mat2(1, 0) + mat1(0, 2) * mat2(2, 0)};
//     std::complex<floatT> prod_02 {mat1(0, 0) * mat2(0, 2) + mat1(0, 1) * mat2(1, 2) + mat1(0, 2) * mat2(2, 2)};
//     std::complex<floatT> prod_20 {mat1(2, 0) * mat2(0, 0) + mat1(2, 1) * mat2(1, 0) + mat1(2, 2) * mat2(2, 0)};
//     std::complex<floatT> prod_22 {mat1(2, 0) * mat2(0, 2) + mat1(2, 1) * mat2(1, 2) + mat1(2, 2) * mat2(2, 2)};
//     std::complex<floatT> temp1   {static_cast<floatT>(0.5) * (prod_00 + std::conj(prod_22))};
//     std::complex<floatT> temp2   {static_cast<floatT>(0.5) * (prod_02 - std::conj(prod_20))};
//     return {temp1, temp2};
// }

// SU2_comp Extract12_new(const Matrix_3x3& mat1, const Matrix_3x3 mat2)
// {
//     std::complex<floatT> prod_11 {mat1(1, 0) * mat2(0, 1) + mat1(1, 1) * mat2(1, 1) + mat1(1, 2) * mat2(2, 1)};
//     std::complex<floatT> prod_12 {mat1(1, 0) * mat2(0, 2) + mat1(1, 1) * mat2(1, 2) + mat1(1, 2) * mat2(2, 2)};
//     std::complex<floatT> prod_21 {mat1(2, 0) * mat2(0, 1) + mat1(2, 1) * mat2(1, 1) + mat1(2, 2) * mat2(2, 1)};
//     std::complex<floatT> prod_22 {mat1(2, 0) * mat2(0, 2) + mat1(2, 1) * mat2(1, 2) + mat1(2, 2) * mat2(2, 2)};
//     std::complex<floatT> temp1   {static_cast<floatT>(0.5) * (prod_11 + std::conj(prod_22))};
//     std::complex<floatT> temp2   {static_cast<floatT>(0.5) * (prod_12 - std::conj(prod_21))};
//     return {temp1, temp2};
// }

#endif // LETTUCE_SU2_HPP
