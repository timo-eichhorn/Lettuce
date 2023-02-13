#ifndef LETTUCE_SU3_HPP
#define LETTUCE_SU3_HPP

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

//----------------------------------------
// TODO: Implement own SU(3) class (e.g. with compression)?
// Provides functions relevant for SU(3) elements like group projections


// TODO: Currently, this might be somehwat inconsistent/unclear: Are the generators the Gell-Mann matrices lambda_j (which are hermitian), or are the generators
//       i * lambda_j (which are algebra elements/antihermitian)? Perhaps we should rename our functions so we can provide both...
// The Gell-Mann matrices lambda_i are traceless hermitian (3x3)-matrices that generate the SU(3) in it's defining representation
// Namely, any group element U may be written as U = exp(i * 0.5 * alpha_j * lambda_j), where alpha_j are eight real coefficients and the summation over j is implicit
namespace SU3::Generators
{
    // The following functions return the unnormalized Gell-Mann matrices
    [[nodiscard]]
    Matrix_3x3 lambda1() noexcept
    {
        Matrix_3x3 tmp;
        tmp << 0.0, 1.0, 0.0,
               1.0, 0.0, 0.0,
               0.0, 0.0, 0.0;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 lambda2() noexcept
    {
        Matrix_3x3 tmp;
        tmp << 0.0      , -i<floatT>, 0.0,
               i<floatT>,  0.0      , 0.0,
               0.0      ,  0.0      , 0.0;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 lambda3() noexcept
    {
        Matrix_3x3 tmp;
        tmp << 1.0,  0.0, 0.0,
               0.0, -1.0, 0.0,
               0.0,  0.0, 0.0;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 lambda4() noexcept
    {
        Matrix_3x3 tmp;
        tmp << 0.0, 0.0, 1.0,
               0.0, 0.0, 0.0,
               1.0, 0.0, 0.0;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 lambda5() noexcept
    {
        Matrix_3x3 tmp;
        tmp << 0.0      , 0.0, -i<floatT>,
               0.0      , 0.0,  0.0,
               i<floatT>, 0.0,  0.0;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 lambda6() noexcept
    {
        Matrix_3x3 tmp;
        tmp << 0.0, 0.0, 0.0,
               0.0, 0.0, 1.0,
               0.0, 1.0, 0.0;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 lambda7() noexcept
    {
        Matrix_3x3 tmp;
        tmp << 0.0, 0.0      ,  0.0,
               0.0, 0.0      , -i<floatT>,
               0.0, i<floatT>,  0.0;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 lambda8() noexcept
    {
        Matrix_3x3 tmp;
        tmp << 1.0, 0.0,  0.0,
               0.0, 1.0,  0.0,
               0.0, 0.0, -2.0;
        return static_cast<floatT>(1.0/std::sqrt(3.0)) * tmp;
    }

    [[nodiscard]]
    Matrix_3x3 lambda(const int i) noexcept
    {
        switch(i)
        {
            case 1:
                return lambda1();
            case 2:
                return lambda2();
            case 3:
                return lambda3();
            case 4:
                return lambda4();
            case 5:
                return lambda5();
            case 6:
                return lambda6();
            case 7:
                return lambda7();
            case 8:
                return lambda8();
            default:
                return Matrix_3x3::Zero();
        }
    }

    // The following functions return exp(i * 0.5 * alpha_j * lambda_j) for a fixed index j (NO summation!)
    [[nodiscard]]
    Matrix_3x3 Exp_lambda1(const floatT alpha_1) noexcept
    {
        floatT       phi   {static_cast<floatT>(0.5) * alpha_1};
        floatT       s_phi {std::sin(phi)};
        floatT       c_phi {std::cos(phi)};
        Matrix_3x3   tmp;
        tmp << c_phi            , i<floatT> * s_phi, 0.0,
               i<floatT> * s_phi, c_phi            , 0.0,
               0.0              , 0.0              , 1.0;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 Exp_lambda2(const floatT alpha_2) noexcept
    {
        floatT       phi   {static_cast<floatT>(0.5) * alpha_2};
        floatT       s_phi {std::sin(phi)};
        floatT       c_phi {std::cos(phi)};
        Matrix_3x3   tmp;
        tmp <<  c_phi, s_phi, 0.0,
               -s_phi, c_phi, 0.0,
                0.0  , 0.0  , 1.0;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 Exp_lambda3(const floatT alpha_3) noexcept
    {
        floatT               phi       {static_cast<floatT>(0.5) * alpha_3};
        std::complex<floatT> exp_i_phi {std::exp(i<floatT> * phi)};
        Matrix_3x3           tmp;
        tmp << exp_i_phi, 0.0                 , 0.0,
               0.0      , std::conj(exp_i_phi), 0.0,
               0.0      , 0.0                 , 1.0;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 Exp_lambda4(const floatT alpha_4) noexcept
    {
        floatT       phi   {static_cast<floatT>(0.5) * alpha_4};
        floatT       s_phi {std::sin(phi)};
        floatT       c_phi {std::cos(phi)};
        Matrix_3x3   tmp;
        tmp << c_phi            , 0.0, i<floatT> * s_phi,
               0.0              , 1.0, 0.0,
               i<floatT> * s_phi, 0.0, c_phi;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 Exp_lambda5(const floatT alpha_5) noexcept
    {
        floatT       phi   {static_cast<floatT>(0.5) * alpha_5};
        floatT       s_phi {std::sin(phi)};
        floatT       c_phi {std::cos(phi)};
        Matrix_3x3   tmp;
        tmp <<  c_phi, 0.0, s_phi,
                0.0  , 1.0, 0.0,
               -s_phi, 0.0, c_phi;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 Exp_lambda6(const floatT alpha_6) noexcept
    {
        floatT       phi   {static_cast<floatT>(0.5) * alpha_6};
        floatT       s_phi {std::sin(phi)};
        floatT       c_phi {std::cos(phi)};
        Matrix_3x3   tmp;
        tmp << 1.0, 0.0              , 0.0,
               0.0, c_phi            , i<floatT> * s_phi,
               0.0, i<floatT> * s_phi, c_phi;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 Exp_lambda7(const floatT alpha_7) noexcept
    {
        floatT       phi   {static_cast<floatT>(0.5) * alpha_7};
        floatT       s_phi {std::sin(phi)};
        floatT       c_phi {std::cos(phi)};
        Matrix_3x3   tmp;
        tmp << 1.0, 0.0  ,  0.0,
               0.0,  c_phi, s_phi,
               0.0, -s_phi, c_phi;
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 Exp_lambda8(const floatT alpha_8) noexcept
    {
        floatT               phi       {static_cast<floatT>(0.5/std::sqrt(3.0)) * alpha_8};
        std::complex<floatT> exp_i_phi {std::exp(i<floatT> * phi)};
        Matrix_3x3           tmp;
        tmp << exp_i_phi, 0.0      , 0.0,
               0.0      , exp_i_phi, 0.0,
               0.0      , 0.0      , 1.0/(exp_i_phi * exp_i_phi);
        return tmp;
    }

    [[nodiscard]]
    Matrix_3x3 Exp_lambda(const int i, const floatT alpha_i) noexcept
    {
        switch(i)
        {
            case 1:
                return Exp_lambda1(alpha_i);
            case 2:
                return Exp_lambda2(alpha_i);
            case 3:
                return Exp_lambda3(alpha_i);
            case 4:
                return Exp_lambda4(alpha_i);
            case 5:
                return Exp_lambda5(alpha_i);
            case 6:
                return Exp_lambda6(alpha_i);
            case 7:
                return Exp_lambda7(alpha_i);
            case 8:
                return Exp_lambda8(alpha_i);
            default:
                return Matrix_3x3::Identity();
        }
    }
} // namespace SU3::Generators

namespace SU3
{
    // Generates a random SU(3) matrix
    [[nodiscard]]
    Matrix_SU3 RandomMat(std::uniform_int_distribution<int>& distribution_choice, std::uniform_real_distribution<floatT>& distribution_unitary)
    {
        Matrix_SU3 tmp;
        int choice {distribution_choice(generator_rand)};
        floatT phi {distribution_unitary(generator_rand)};

        switch(choice)
        {
            case 1:
            {
                floatT s_phi {std::sin(phi)};
                floatT c_phi {std::cos(phi)};
                tmp << c_phi            , i<floatT> * s_phi, 0.0,
                       i<floatT> * s_phi, c_phi            , 0.0,
                       0.0              , 0.0              , 1.0;
            }
            break;
            case 2:
            {
                floatT s_phi {std::sin(phi)};
                floatT c_phi {std::cos(phi)};
                tmp <<  c_phi, s_phi, 0.0,
                       -s_phi, c_phi, 0.0,
                        0.0  , 0.0  , 1.0;
            }
            break;
            case 3:
            {
                std::complex<floatT> exp_i_phi {std::exp(i<floatT> * phi)};
                tmp << exp_i_phi, 0.0                 , 0.0,
                       0.0      , std::conj(exp_i_phi), 0.0,
                       0.0      , 0.0                 , 1.0;
            }
            break;
            case 4:
            {
                floatT s_phi {std::sin(phi)};
                floatT c_phi {std::cos(phi)};
                tmp << c_phi            , 0.0, i<floatT> * s_phi,
                       0.0              , 1.0, 0.0,
                       i<floatT> * s_phi, 0.0, c_phi;
            }
            break;
            case 5:
            {
                floatT s_phi {std::sin(phi)};
                floatT c_phi {std::cos(phi)};
                tmp <<  c_phi, 0.0, s_phi,
                        0.0  , 1.0, 0.0,
                       -s_phi, 0.0, c_phi;
            }
            break;
            case 6:
            {
                floatT s_phi {std::sin(phi)};
                floatT c_phi {std::cos(phi)};
                tmp << 1.0, 0.0              , 0.0,
                       0.0, c_phi            , i<floatT> * s_phi,
                       0.0, i<floatT> * s_phi, c_phi;
            }
            break;
            case 7:
            {
                floatT s_phi {std::sin(phi)};
                floatT c_phi {std::cos(phi)};
                tmp << 1.0, 0.0  ,  0.0,
                       0.0,  c_phi, s_phi,
                       0.0, -s_phi, c_phi;
            }
            break;
            case 8:
            {
                floatT               phi_tilde {phi / static_cast<floatT>(std::sqrt(3))};
                std::complex<floatT> exp_i_phi {std::exp(i<floatT> * phi_tilde)};
                tmp << exp_i_phi, 0.0      , 0.0,
                       0.0      , exp_i_phi, 0.0,
                       0.0      , 0.0      , 1.0/(exp_i_phi * exp_i_phi);
            }
            break;
        }
        return tmp;
    }

    // Generates a random SU(3) matrix
    [[nodiscard]]
    Matrix_SU3 RandomMatParallel(const int choice, const floatT phi)
    {
        Matrix_SU3 tmp;

        switch(choice)
        {
            case 1:
            {
                floatT s_phi {std::sin(phi)};
                floatT c_phi {std::cos(phi)};
                tmp << c_phi            , i<floatT> * s_phi, 0.0,
                       i<floatT> * s_phi, c_phi            , 0.0,
                       0.0              , 0.0              , 1.0;
            }
            break;
            case 2:
            {
                floatT s_phi {std::sin(phi)};
                floatT c_phi {std::cos(phi)};
                tmp <<  c_phi, s_phi, 0.0,
                       -s_phi, c_phi, 0.0,
                        0.0  , 0.0  , 1.0;
            }
            break;
            case 3:
            {
                std::complex<floatT> exp_i_phi {std::exp(i<floatT> * phi)};
                tmp << exp_i_phi, 0.0                 , 0.0,
                       0.0      , std::conj(exp_i_phi), 0.0,
                       0.0      , 0.0                 , 1.0;
            }
            break;
            case 4:
            {
                floatT s_phi {std::sin(phi)};
                floatT c_phi {std::cos(phi)};
                tmp << c_phi            , 0.0, i<floatT> * s_phi,
                       0.0              , 1.0, 0.0,
                       i<floatT> * s_phi, 0.0, c_phi;
            }
            break;
            case 5:
            {
                floatT s_phi {std::sin(phi)};
                floatT c_phi {std::cos(phi)};
                tmp <<  c_phi, 0.0, s_phi,
                        0.0  , 1.0, 0.0,
                       -s_phi, 0.0, c_phi;
            }
            break;
            case 6:
            {
                floatT s_phi {std::sin(phi)};
                floatT c_phi {std::cos(phi)};
                tmp << 1.0, 0.0              , 0.0,
                       0.0, c_phi            , i<floatT> * s_phi,
                       0.0, i<floatT> * s_phi, c_phi;
            }
            break;
            case 7:
            {
                floatT s_phi {std::sin(phi)};
                floatT c_phi {std::cos(phi)};
                tmp << 1.0, 0.0  ,  0.0,
                       0.0,  c_phi, s_phi,
                       0.0, -s_phi, c_phi;
            }
            break;
            case 8:
            {
                floatT               phi_tilde {phi / static_cast<floatT>(std::sqrt(3))};
                std::complex<floatT> exp_i_phi {std::exp(i<floatT> * phi_tilde)};
                tmp << exp_i_phi, 0.0      , 0.0,
                       0.0      , exp_i_phi, 0.0,
                       0.0      , 0.0      , 1.0/(exp_i_phi * exp_i_phi);
            }
            break;
        }
        return tmp;
    }
} // namespace SU3

namespace SU3::Projection
{

    // Used to restore full SU(3) matrix from first two columns
    // __attribute__((always_inline))
    void RestoreLastColumn(Matrix_SU3& Mat) noexcept
    {
        Mat(0, 2) = std::conj(Mat(1, 0) * Mat(2, 1) - Mat(2, 0) * Mat(1, 1));
        Mat(1, 2) = std::conj(Mat(2, 0) * Mat(0, 1) - Mat(0, 0) * Mat(2, 1));
        Mat(2, 2) = std::conj(Mat(0, 0) * Mat(1, 1) - Mat(1, 0) * Mat(0, 1));
    }

    void RestoreLastRow(Matrix_SU3& Mat) noexcept
    {
        Mat(2, 0) = std::conj(Mat(0, 1) * Mat(1, 2) - Mat(0, 2) * Mat(1, 1));
        Mat(2, 1) = std::conj(Mat(0, 2) * Mat(1, 0) - Mat(0, 0) * Mat(1, 2));
        Mat(2, 2) = std::conj(Mat(0, 0) * Mat(1, 1) - Mat(0, 1) * Mat(1, 0));
    }

    // Projects a single link back on SU(3) via Gram-Schmidt
    void GramSchmidt(Matrix_SU3& Mat) noexcept
    {
        // floatT norm0 {static_cast<floatT>(1.0)/std::sqrt(std::real(Mat(0, 0) * std::conj(Mat(0, 0)) + Mat(0, 1) * std::conj(Mat(0, 1)) + Mat(0, 2) * std::conj(Mat(0, 2))))};
        // for (int n = 0; n < 3; ++n)
        // {
        //     Mat(0, n) = norm0 * Mat(0, n);
        // }
        // std::complex<floatT> psi {Mat(1, 0) * std::conj(Mat(0, 0)) + Mat(1, 1) * std::conj(Mat(0, 1)) + Mat(1, 2) * std::conj(Mat(0, 2))};
        // for (int n = 0; n < 3; ++n)
        // {
        //     Mat(1, n) =  Mat(1, n) - psi * Mat(0, n);
        // }
        // floatT norm1 {static_cast<floatT>(1.0)/std::sqrt(std::real(Mat(1, 0) * std::conj(Mat(1, 0)) + Mat(1, 1) * std::conj(Mat(1, 1)) + Mat(1, 2) * std::conj(Mat(1, 2))))};
        // for (int n = 0; n < 3; ++n)
        // {
        //     Mat(1, n) = norm1 * Mat(1, n);
        // }
        // Mat(2, 0) = std::conj(Mat(0, 1) * Mat(1, 2) - Mat(0, 2) * Mat(1, 1));
        // Mat(2, 1) = std::conj(Mat(0, 2) * Mat(1, 0) - Mat(0, 0) * Mat(1, 2));
        // Mat(2, 2) = std::conj(Mat(0, 0) * Mat(1, 1) - Mat(0, 1) * Mat(1, 0));
        // Per default, Eigen uses a column-major storage order, so the first index is the inner index in terms of the memory layout
        floatT norm0 {static_cast<floatT>(1.0)/std::sqrt(std::real(Mat(0, 0) * std::conj(Mat(0, 0)) + Mat(1, 0) * std::conj(Mat(1, 0)) + Mat(2, 0) * std::conj(Mat(2, 0))))};
        for (int n = 0; n < 3; ++n)
        {
            Mat(n, 0) = norm0 * Mat(n, 0);
        }
        std::complex<floatT> psi {Mat(0, 1) * std::conj(Mat(0, 0)) + Mat(1, 1) * std::conj(Mat(1, 0)) + Mat(2, 1) * std::conj(Mat(2, 0))};
        for (int n = 0; n < 3; ++n)
        {
            Mat(n, 1) = Mat(n, 1) - psi * Mat(n, 0);
        }
        floatT norm1 {static_cast<floatT>(1.0)/std::sqrt(std::real(Mat(0, 1) * std::conj(Mat(0, 1)) + Mat(1, 1) * std::conj(Mat(1, 1)) + Mat(2, 1) * std::conj(Mat(2, 1))))};
        for (int n = 0; n < 3; ++n)
        {
            Mat(n, 1) = norm1 * Mat(n, 1);
        }
        Mat(0, 2) = std::conj(Mat(1, 0) * Mat(2, 1) - Mat(2, 0) * Mat(1, 1));
        Mat(1, 2) = std::conj(Mat(2, 0) * Mat(0, 1) - Mat(0, 0) * Mat(2, 1));
        Mat(2, 2) = std::conj(Mat(0, 0) * Mat(1, 1) - Mat(1, 0) * Mat(0, 1));
        // TODO: If RestoreLastColumn() is declared as an inline function, we get the same results as using the lines above,
        //       but if it is not declared as inline, we get different results. Huhh???
        // RestoreLastColumn(Mat);
    }

    //-----
    // Projects a single link back on SU(3) via Kenney-Laub iteration (used for direct SU(N) overrelaxation updates)

    void KenneyLaub(Matrix_SU3& Mat) noexcept
    {
        // Based on section 3 in arXiv:1701.00726
        // Use Kenney-Laub iteration to get the unitary part of the polar decomposition of the matrix
        // TODO: Reverse order of polar decomposition and determinant normalization?
        //       => First normalize with determinant, then iterate?
        Matrix_SU3 X;
        do
        {
            X = Mat.adjoint() * Mat;
            Mat = Mat/static_cast<floatT>(3.0) * (Matrix_SU3::Identity() + static_cast<floatT>(8.0/3.0) * (Mat.adjoint() * Mat + static_cast<floatT>(1.0/3.0) * Matrix_SU3::Identity()).inverse());
        }
        // Iterate as long as the Frobenius norm of the difference between the unit matrix and X = Mat.adjoint() * Mat is greater than 1e-6
        while ((Matrix_SU3::Identity() - X).norm() > static_cast<floatT>(1e-6));
        // Then normalize to set determinant equal to 1
        Mat = Mat * (static_cast<floatT>(1.0)/std::pow(Mat.determinant(), static_cast<floatT>(1.0/3.0)));
        // TODO: Use std::cbrt instead of std::pow? std::cbrt doesn't seem to work on std::complex<T>...
        // Mat = Mat * (1.f/std::cbrt(Mat.determinant()));
    }

    //-----
    // Projects a single element onto su(3) algebra (traceless antihermitian)

    [[nodiscard]]
    Matrix_3x3 Algebra(const Matrix_3x3& Mat) noexcept
    {
        return static_cast<floatT>(0.5) * (Mat - Mat.adjoint()) - static_cast<floatT>(1.0/6.0) * (Mat - Mat.adjoint()).trace() * Matrix_3x3::Identity();
    }

    [[nodiscard]]
    Matrix_3x3 TracelessAntihermitian(const Matrix_3x3& Mat) noexcept
    {
        return static_cast<floatT>(0.5) * (Mat - Mat.adjoint()) - static_cast<floatT>(1.0/6.0) * (Mat - Mat.adjoint()).trace() * Matrix_3x3::Identity();
    }

    // Projects a single element onto traceless hermitian matrix

    [[nodiscard]]
    Matrix_3x3 TracelessHermitian(const Matrix_3x3& Mat) noexcept
    {
        return static_cast<floatT>(0.5) * (Mat + Mat.adjoint()) - static_cast<floatT>(1.0/6.0) * (Mat + Mat.adjoint()).trace() * Matrix_3x3::Identity();
    }
} // namespace SU3::Projection

namespace SU3::Tests
{
    //-----
    // Test unitarity of a matrix (up to a certain precision)

    [[nodiscard]]
    bool Unitarity(const Matrix_SU3& Mat, const floatT prec = 1e-6) noexcept
    {
        // By default .norm() uses the Frobenius norm, i.e., the square root of the sum of all squared matrix entries
        if ((Matrix_SU3::Identity() - Mat.adjoint() * Mat).norm() > prec)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    [[nodiscard]]
    bool UnitarityAll(const GaugeField& U, const floatT prec = 1e-6) noexcept
    {
        bool IsUnitary {true};
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            if (not Unitarity(U({t, x, y, z, mu}), prec))
            {
                IsUnitary = false;
                return IsUnitary;
            }
        }
        return IsUnitary;
    }

    //-----
    // Test how close the determinant of a matrix is to 1 (up to a certain precision)

    [[nodiscard]]
    bool Special(const Matrix_SU3& Mat, const floatT prec = 1e-6) noexcept
    {
        if (std::abs(Mat.determinant() - static_cast<floatT>(1.0)) > prec)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    [[nodiscard]]
    bool SpecialAll(const GaugeField& U, const floatT prec = 1e-6) noexcept
    {
        bool IsSpecial {true};
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            if (not Special(U({t, x, y, z, mu}), prec))
            {
                IsSpecial = false;
                return IsSpecial;
            }
        }
        return IsSpecial;
    }

    //-----
    // Test if a matrix is a SU(3) group element (up to a certain precision)

    [[nodiscard]]
    bool TestSU3(const Matrix_SU3& Mat, const floatT prec = 1e-6) noexcept
    {
        if ((Matrix_SU3::Identity() - Mat.adjoint() * Mat).norm() > prec or std::abs(Mat.determinant() - static_cast<floatT>(1.0)) > prec)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    [[nodiscard]]
    bool TestSU3All(const GaugeField& U, const floatT prec = 1e-6) noexcept
    {
        bool InGroup {true};
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            if (not TestSU3(U({t, x, y, z, mu}), prec))
            {
                InGroup = false;
                return InGroup;
            }
        }
        return InGroup;
    }

    //-----
    // Test if a matrix is a su(3) algebra element (up to a certain precision)
    // NOTE: Technically this checks if a matrix is traceless and hermitian, while su(3) matrices are anti-hermitian

    [[nodiscard]]
    bool Testsu3(const Matrix_SU3& Mat, const floatT prec = 1e-6) noexcept
    {
        if ((Mat - Mat.adjoint()).norm() > prec or std::abs(Mat.trace()) > prec)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    // NOTE: Technically this checks if a matrix is traceless and hermitian, while su(3) matrices are anti-hermitian

    [[nodiscard]]
    bool Testsu3All(const GaugeField& U, const floatT prec = 1e-6) noexcept
    {
        bool InAlgebra {true};
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            if (not Testsu3(U({t, x, y, z, mu}), prec))
            {
                InAlgebra = false;
                return InAlgebra;
            }
        }
        return InAlgebra;
    }
} // namespace SU3::Tests

#endif // LETTUCE_SU3_HPP
