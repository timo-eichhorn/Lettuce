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
    // Projects a single link back on SU(3) via Gram-Schmidt
    void GramSchmidt(Matrix_SU3& GluonMatrix) noexcept
    {
        floatT norm0 {static_cast<floatT>(1.0)/std::sqrt(std::real(GluonMatrix(0, 0) * conj(GluonMatrix(0, 0)) + GluonMatrix(0, 1) * conj(GluonMatrix(0, 1)) + GluonMatrix(0, 2) * conj(GluonMatrix(0, 2))))};
        for (int n = 0; n < 3; ++n)
        {
            GluonMatrix(0, n) = norm0 * GluonMatrix(0, n);
        }
        std::complex<floatT> psi {GluonMatrix(1, 0) * conj(GluonMatrix(0, 0)) + GluonMatrix(1, 1) * conj(GluonMatrix(0, 1)) + GluonMatrix(1, 2) * conj(GluonMatrix(0, 2))};
        for (int n = 0; n < 3; ++n)
        {
            GluonMatrix(1, n) =  GluonMatrix(1, n) - psi * GluonMatrix(0, n);
        }
        floatT norm1 {static_cast<floatT>(1.0)/std::sqrt(std::real(GluonMatrix(1, 0) * conj(GluonMatrix(1, 0)) + GluonMatrix(1, 1) * conj(GluonMatrix(1, 1)) + GluonMatrix(1, 2) * conj(GluonMatrix(1, 2))))};
        for (int n = 0; n < 3; ++n)
        {
            GluonMatrix(1, n) = norm1 * GluonMatrix(1, n);
        }
        GluonMatrix(2, 0) = conj(GluonMatrix(0, 1) * GluonMatrix(1, 2) - GluonMatrix(0, 2) * GluonMatrix(1, 1));
        GluonMatrix(2, 1) = conj(GluonMatrix(0, 2) * GluonMatrix(1, 0) - GluonMatrix(0, 0) * GluonMatrix(1, 2));
        GluonMatrix(2, 2) = conj(GluonMatrix(0, 0) * GluonMatrix(1, 1) - GluonMatrix(0, 1) * GluonMatrix(1, 0));
    }

    //-----
    // Projects matrices back on SU(3) via Gram-Schmidt

    // void ProjectionSU3(GaugeField4D<Matrix_SU3>& Gluon)
    // {
    //     for (auto& ind0 : Gluon)
    //     for (auto& ind1 : ind0)
    //     for (auto& ind2 : ind1)
    //     for (auto& ind3 : ind2)
    //     for (auto& GluonMatrix : ind3)
    //     {
    //         floatT norm0 {static_cast<floatT>(1.0)/std::sqrt(std::real(GluonMatrix(0, 0) * conj(GluonMatrix(0, 0)) + GluonMatrix(0, 1) * conj(GluonMatrix(0, 1)) + GluonMatrix(0, 2) * conj(GluonMatrix(0, 2))))};
    //         for (int n = 0; n < 3; ++n)
    //         {
    //             GluonMatrix(0, n) = norm0 * GluonMatrix(0, n);
    //         }
    //         std::complex<floatT> psi {GluonMatrix(1, 0) * conj(GluonMatrix(0, 0)) + GluonMatrix(1, 1) * conj(GluonMatrix(0, 1)) + GluonMatrix(1, 2) * conj(GluonMatrix(0, 2))};
    //         for (int n = 0; n < 3; ++n)
    //         {
    //             GluonMatrix(1, n) =  GluonMatrix(1, n) - psi * GluonMatrix(0, n);
    //         }
    //         floatT norm1 {static_cast<floatT>(1.0)/std::sqrt(std::real(GluonMatrix(1, 0) * conj(GluonMatrix(1, 0)) + GluonMatrix(1, 1) * conj(GluonMatrix(1, 1)) + GluonMatrix(1, 2) * conj(GluonMatrix(1, 2))))};
    //         for (int n = 0; n < 3; ++n)
    //         {
    //             GluonMatrix(1, n) = norm1 * GluonMatrix(1, n);
    //         }
    //         GluonMatrix(2, 0) = conj(GluonMatrix(0, 1) * GluonMatrix(1, 2) - GluonMatrix(0, 2) * GluonMatrix(1, 1));
    //         GluonMatrix(2, 1) = conj(GluonMatrix(0, 2) * GluonMatrix(1, 0) - GluonMatrix(0, 0) * GluonMatrix(1, 2));
    //         GluonMatrix(2, 2) = conj(GluonMatrix(0, 0) * GluonMatrix(1, 1) - GluonMatrix(0, 1) * GluonMatrix(1, 0));
    //     }
    // }

    //-----
    // Projects a single link back on SU(3) via Kenney-Laub iteration (used for direct SU(N) overrelaxation updates)

    void KenneyLaub(Matrix_SU3& GluonMatrix) noexcept
    {
        // Based on section 3 in arXiv:1701.00726
        // Use Kenney-Laub iteration to get the unitary part of the polar decomposition of the matrix
        // TODO: Reverse order of polar decomposition and determinant normalization?
        //       => First normalize with determinant, then iterate?
        Matrix_SU3 X;
        do
        {
            X = GluonMatrix.adjoint() * GluonMatrix;
            GluonMatrix = GluonMatrix/static_cast<floatT>(3.0) * (Matrix_SU3::Identity() + static_cast<floatT>(8.0/3.0) * (GluonMatrix.adjoint() * GluonMatrix + static_cast<floatT>(1.0/3.0) * Matrix_SU3::Identity()).inverse());
        }
        // Iterate as long as the Frobenius norm of the difference between the unit matrix and X = GluonMatrix.adjoint() * GluonMatrix is greater than 1e-6
        while ((Matrix_SU3::Identity() - X).norm() > static_cast<floatT>(1e-6));
        // Then normalize to set determinant equal to 1
        GluonMatrix = GluonMatrix * (static_cast<floatT>(1.0)/std::pow(GluonMatrix.determinant(), static_cast<floatT>(1.0/3.0)));
        // TODO: Use std::cbrt instead of std::pow? std::cbrt doesn't seem to work on std::complex<T>...
        // GluonMatrix = GluonMatrix * (1.f/std::cbrt(GluonMatrix.determinant()));
    }

    //-----
    // Projects a single element onto su(3) algebra (traceless antihermitian)

    Matrix_3x3 Algebra(const Matrix_3x3& mat) noexcept
    {
        return static_cast<floatT>(0.5) * (mat - mat.adjoint()) - static_cast<floatT>(1.0/6.0) * (mat - mat.adjoint()).trace() * Matrix_3x3::Identity();
    }

    Matrix_3x3 TracelessAntihermitian(const Matrix_3x3& mat) noexcept
    {
        return static_cast<floatT>(0.5) * (mat - mat.adjoint()) - static_cast<floatT>(1.0/6.0) * (mat - mat.adjoint()).trace() * Matrix_3x3::Identity();
    }

    // Projects a single element onto traceless hermitian matrix

    Matrix_3x3 TracelessHermitian(const Matrix_3x3& mat) noexcept
    {
        return static_cast<floatT>(0.5) * (mat + mat.adjoint()) - static_cast<floatT>(1.0/6.0) * (mat + mat.adjoint()).trace() * Matrix_3x3::Identity();
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
    bool UnitarityAll(const GaugeField& Gluon, const floatT prec = 1e-6) noexcept
    {
        bool IsUnitary {true};
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            if (not Unitarity(Gluon({t, x, y, z, mu}), prec))
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
    bool SpecialAll(const GaugeField& Gluon, const floatT prec = 1e-6) noexcept
    {
        bool IsSpecial {true};
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            if (not Special(Gluon({t, x, y, z, mu}), prec))
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
    bool TestSU3All(const GaugeField& Gluon, const floatT prec = 1e-6) noexcept
    {
        bool InGroup {true};
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            if (not TestSU3(Gluon({t, x, y, z, mu}), prec))
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
    bool Testsu3All(const GaugeField& Gluon, const floatT prec = 1e-6) noexcept
    {
        bool InAlgebra {true};
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        for (int mu = 0; mu < 4; ++mu)
        {
            if (not Testsu3(Gluon({t, x, y, z, mu}), prec))
            {
                InAlgebra = false;
                return InAlgebra;
            }
        }
        return InAlgebra;
    }
} // namespace SU3::Tests

#endif // LETTUCE_SU3_HPP
