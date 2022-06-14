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
    // Projects a single element onto su(3) algebra

    Matrix_3x3 Algebra(const Matrix_3x3& mat)
    {
        return static_cast<floatT>(0.5) * (mat + mat.adjoint()) - static_cast<floatT>(1.0/6.0) * (mat + mat.adjoint()).trace() * Matrix_3x3::Identity();
    }
}

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
            if (Unitarity(Gluon({t, x, y, z, mu}), prec) != true)
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
            if (Special(Gluon({t, x, y, z, mu}), prec) != true)
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
        if ((Matrix_SU3::Identity() - Mat.adjoint() * Mat).norm() > prec && std::abs(Mat.determinant() - static_cast<floatT>(1.0)) > prec)
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
            if (TestSU3(Gluon({t, x, y, z, mu}), prec) != true)
            {
                InGroup = false;
                return InGroup;
            }
        }
        return InGroup;
    }

    //-----
    // Test if a matrix is a su(3) algebra element (up to a certain precision)

    [[nodiscard]]
    bool Testsu3(const Matrix_SU3& Mat, const floatT prec = 1e-6) noexcept
    {
        if ((Mat - Mat.adjoint()).norm() > prec && std::abs(Mat.trace()) > prec)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

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
            if (Testsu3(Gluon({t, x, y, z, mu}), prec) != true)
            {
                InAlgebra = false;
                return InAlgebra;
            }
        }
        return InAlgebra;
    }
}

#endif // LETTUCE_SU3_HPP
