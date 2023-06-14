#ifndef LETTUCE_CUSTOM_SU3_HPP // TODO: Change this to match filename later on
#define LETTUCE_CUSTOM_SU3_HPP

template<typename floatT>
class Matrix_3x3_
{
    using complexT = std::complex<floatT>;
    public:
        // "A static data member is not part of the subobjects of a class" [class.static.data]
        static constexpr int Nrow      {3};
        static constexpr int Ncolumn   {3};
        static constexpr int Nelements {Nrow * Ncolumn};
        static constexpr int Ncolor    {3};
        // Matrix_3x3_ identity() {
        //     Matrix_3x3_ tmp;
        //     tmp[0]
        // } 
        // std::array<complexT, Ncol> Mat;
        // Standard C++ order/row-major:
        // complexT e11, e12, e13, e21, e22, e23, e31, e32, e33;
        // Eigen order/column-major:
        // complexT e11, e21, e31, e12, e22, e32, e13, e23, e33;
        std::array<complexT, Nelements> data;

        Matrix_3x3_() noexcept = default;

        // TODO: In which order should the constructor accept the arguments? Memory layout, or layout on paper?
        Matrix_3x3_(const complexT e11_in, const complexT e12_in, const complexT e13_in,
                    const complexT e21_in, const complexT e22_in, const complexT e23_in,
                    const complexT e31_in, const complexT e32_in, const complexT e33_in) noexcept :
        // Note that the members are initialized in the same order as they are declared above, not in the order that they appear in the initializer list below
        // Standard C++ order/row-major:
        // data({e11_in, e12_in, e13_in, e21_in, e22_in, e23_in, e31_in, e32_in, e33_in})
        // Eigen order/column-major:
        data({e11_in, e21_in, e31_in, e12_in, e22_in, e32_in, e13_in, e23_in, e33_in})
        // e11(e11_in), e12(e12_in), e13(e13_in),
        // e21(e21_in), e22(e22_in), e23(e23_in),
        // e31(e31_in), e32(e32_in), e33(e33_in)
        {}

        // Matrix_3x3_(const std::array<complexT, 9>& Mat_in) noexcept :
        // e11(Mat_in[0]), e12(Mat_in[1]), e13(Mat_in[2]),
        // e21(Mat_in[3]), e22(Mat_in[4]), e23(Mat_in[5]),
        // e31(Mat_in[6]), e32(Mat_in[7]), e33(Mat_in[8])
        // {}

        // Delete initializer list constructor
        // template<typename T>
        // Matrix_3x3_(std::initializer_list<T>) = delete;

        Matrix_3x3_& operator=(const Matrix_3x3_& Mat) noexcept
        {
            // Old
            // e11 = Mat.e11; e12 = Mat.e12; e13 = Mat.e13;
            // e21 = Mat.e21; e22 = Mat.e22; e23 = Mat.e23;
            // e31 = Mat.e31; e32 = Mat.e32; e33 = Mat.e33;
            // New
            // e11 = Mat.e11; e21 = Mat.e21; e31 = Mat.e31;
            // e12 = Mat.e12; e22 = Mat.e22; e32 = Mat.e32;
            // e13 = Mat.e13; e23 = Mat.e23; e33 = Mat.e33;
            // #pragma omp simd
            for (int i = 0; i < Nelements; ++i)
            {
                data[i] = Mat.data[i];
            }
            return *this;
        }

        // Delete initializer list copy constructor
        template<typename T>
        Matrix_3x3_& operator=(std::initializer_list<T>) = delete;

        // TODO: Problematic, compiler complains about ambiguity between unary and binary operator -
        friend Matrix_3x3_ operator-(const Matrix_3x3_& Mat) noexcept
        {
            return {-Mat.data[0], -Mat.data[3], -Mat.data[6],
                    -Mat.data[1], -Mat.data[4], -Mat.data[7],
                    -Mat.data[2], -Mat.data[5], -Mat.data[8],};
        }

        friend Matrix_3x3_& operator+=(Matrix_3x3_& Mat1, const Matrix_3x3_& Mat2) noexcept
        {
            // Old
            // Mat1.e11 += Mat2.e11; Mat1.e12 += Mat2.e12; Mat1.e13 += Mat2.e13;
            // Mat1.e21 += Mat2.e21; Mat1.e22 += Mat2.e22; Mat1.e23 += Mat2.e23;
            // Mat1.e31 += Mat2.e31; Mat1.e32 += Mat2.e32; Mat1.e33 += Mat2.e33;
            // New
            // Mat1.e11 += Mat2.e11; Mat1.e21 += Mat2.e21; Mat1.e31 += Mat2.e31;
            // Mat1.e12 += Mat2.e12; Mat1.e22 += Mat2.e22; Mat1.e32 += Mat2.e32;
            // Mat1.e13 += Mat2.e13; Mat1.e23 += Mat2.e23; Mat1.e33 += Mat2.e33;
            // #pragma omp simd
            for (int i = 0; i < Nelements; ++i)
            {
                Mat1.data[i] += Mat2.data[i];
            }
            return Mat1;
        }

        friend Matrix_3x3_& operator-=(Matrix_3x3_& Mat1, const Matrix_3x3_& Mat2) noexcept
        {
            // Old
            // Mat1.e11 -= Mat2.e11; Mat1.e12 -= Mat2.e12; Mat1.e13 -= Mat2.e13;
            // Mat1.e21 -= Mat2.e21; Mat1.e22 -= Mat2.e22; Mat1.e23 -= Mat2.e23;
            // Mat1.e31 -= Mat2.e31; Mat1.e32 -= Mat2.e32; Mat1.e33 -= Mat2.e33;
            // New
            // Mat1.e11 -= Mat2.e11; Mat1.e21 -= Mat2.e21; Mat1.e31 -= Mat2.e31;
            // Mat1.e12 -= Mat2.e12; Mat1.e22 -= Mat2.e22; Mat1.e32 -= Mat2.e32;
            // Mat1.e13 -= Mat2.e13; Mat1.e23 -= Mat2.e23; Mat1.e33 -= Mat2.e33;
            // #pragma omp simd
            for (int i = 0; i < Nelements; ++i)
            {
                Mat1.data[i] -= Mat2.data[i];
            }
            return Mat1;
        }

        // Scalar multiplication
        // TODO: Do we need two versions for real and comlex types?

        friend Matrix_3x3_& operator*=(Matrix_3x3_& Mat, const floatT a) noexcept
        {
            // Old
            // Mat.e11 *= a; Mat.e12 *= a; Mat.e13 *= a;
            // Mat.e21 *= a; Mat.e22 *= a; Mat.e23 *= a;
            // Mat.e31 *= a; Mat.e32 *= a; Mat.e33 *= a;
            // New
            // Mat.e11 *= a; Mat.e21 *= a; Mat.e31 *= a;
            // Mat.e12 *= a; Mat.e22 *= a; Mat.e32 *= a;
            // Mat.e13 *= a; Mat.e23 *= a; Mat.e33 *= a;
            // #pragma omp simd
            for (int i = 0; i < Nelements; ++i)
            {
                Mat.data[i] *= a;
            }
            return Mat;
        }

        friend Matrix_3x3_& operator*=(Matrix_3x3_& Mat, const complexT a) noexcept
        {
            // Old
            // Mat.e11 *= a; Mat.e12 *= a; Mat.e13 *= a;
            // Mat.e21 *= a; Mat.e22 *= a; Mat.e23 *= a;
            // Mat.e31 *= a; Mat.e32 *= a; Mat.e33 *= a;
            // New
            // Mat.e11 *= a; Mat.e21 *= a; Mat.e31 *= a;
            // Mat.e12 *= a; Mat.e22 *= a; Mat.e32 *= a;
            // Mat.e13 *= a; Mat.e23 *= a; Mat.e33 *= a;
            // #pragma omp simd
            for (int i = 0; i < Nelements; ++i)
            {
                Mat.data[i] *= a;
            }
            return Mat;
        }

        // friend Matrix_3x3_& operator/=(Matrix_3x3_& Mat, const floatT a) noexcept
        // {
        //     Mat.e11 /= a; Mat.e12 /= a; Mat.e13 /= a;
        //     Mat.e21 /= a; Mat.e22 /= a; Mat.e23 /= a;
        //     Mat.e31 /= a; Mat.e32 /= a; Mat.e33 /= a;
        //     return Mat;
        // }

        // friend Matrix_3x3_& operator/=(Matrix_3x3_& Mat, const complexT a) noexcept
        // {
        //     Mat.e11 /= a; Mat.e12 /= a; Mat.e13 /= a;
        //     Mat.e21 /= a; Mat.e22 /= a; Mat.e23 /= a;
        //     Mat.e31 /= a; Mat.e32 /= a; Mat.e33 /= a;
        //     return Mat;
        // }

        friend Matrix_3x3_ operator*(const Matrix_3x3_& Mat, const floatT a) noexcept
        {
            Matrix_3x3_ tmp {Mat};
            return tmp *= a;
        }

        friend Matrix_3x3_ operator*(const floatT a, const Matrix_3x3_& Mat) noexcept
        {
            Matrix_3x3_ tmp {Mat};
            return tmp *= a;
        }

        friend Matrix_3x3_ operator*(const Matrix_3x3_& Mat, const complexT a) noexcept
        {
            Matrix_3x3_ tmp {Mat};
            return tmp *= a;
        }

        friend Matrix_3x3_ operator*(const complexT a, const Matrix_3x3_& Mat) noexcept
        {
            Matrix_3x3_ tmp {Mat};
            return tmp *= a;
        }

        friend Matrix_3x3_ operator/(const Matrix_3x3_& Mat, const floatT a) noexcept
        {
            Matrix_3x3_ tmp {Mat};
            return tmp /= a;
        }

        friend Matrix_3x3_ operator/(const floatT a, const Matrix_3x3_& Mat) noexcept
        {
            Matrix_3x3_ tmp {Mat};
            return tmp /= a;
        }

        friend Matrix_3x3_ operator/(const Matrix_3x3_& Mat, const complexT a) noexcept
        {
            Matrix_3x3_ tmp {Mat};
            return tmp /= a;
        }

        friend Matrix_3x3_ operator/(const complexT a, const Matrix_3x3_& Mat) noexcept
        {
            Matrix_3x3_ tmp {Mat};
            return tmp /= a;
        }

        // Matrix multiplication

        friend Matrix_3x3_& operator*=(Matrix_3x3_& Mat1, const Matrix_3x3_& Mat2) noexcept
        {
            // Old
            // complexT tmp_e11 {Mat1.e11 * Mat2.e11 + Mat1.e12 * Mat2.e21 + Mat1.e13 * Mat2.e31};
            // complexT tmp_e12 {Mat1.e11 * Mat2.e12 + Mat1.e12 * Mat2.e22 + Mat1.e13 * Mat2.e32};
            // complexT tmp_e13 {Mat1.e11 * Mat2.e13 + Mat1.e12 * Mat2.e23 + Mat1.e13 * Mat2.e33};
            // complexT tmp_e21 {Mat1.e21 * Mat2.e11 + Mat1.e22 * Mat2.e21 + Mat1.e23 * Mat2.e31};
            // complexT tmp_e22 {Mat1.e21 * Mat2.e12 + Mat1.e22 * Mat2.e22 + Mat1.e23 * Mat2.e32};
            // complexT tmp_e23 {Mat1.e21 * Mat2.e13 + Mat1.e22 * Mat2.e23 + Mat1.e23 * Mat2.e33};
            // complexT tmp_e31 {Mat1.e31 * Mat2.e11 + Mat1.e32 * Mat2.e21 + Mat1.e33 * Mat2.e31};
            // complexT tmp_e32 {Mat1.e31 * Mat2.e12 + Mat1.e32 * Mat2.e22 + Mat1.e33 * Mat2.e32};
            // complexT tmp_e33 {Mat1.e31 * Mat2.e13 + Mat1.e32 * Mat2.e23 + Mat1.e33 * Mat2.e33};
            // Mat1.e11 = tmp_e11; Mat1.e12 = tmp_e12; Mat1.e13 = tmp_e13;
            // Mat1.e21 = tmp_e21; Mat1.e22 = tmp_e22; Mat1.e23 = tmp_e23;
            // Mat1.e31 = tmp_e31; Mat1.e32 = tmp_e32; Mat1.e33 = tmp_e33;
            // New
            // complexT tmp_e11 {Mat1.e11 * Mat2.e11 + Mat1.e12 * Mat2.e21 + Mat1.e13 * Mat2.e31};
            // complexT tmp_e21 {Mat1.e21 * Mat2.e11 + Mat1.e22 * Mat2.e21 + Mat1.e23 * Mat2.e31};
            // complexT tmp_e31 {Mat1.e31 * Mat2.e11 + Mat1.e32 * Mat2.e21 + Mat1.e33 * Mat2.e31};
            // complexT tmp_e12 {Mat1.e11 * Mat2.e12 + Mat1.e12 * Mat2.e22 + Mat1.e13 * Mat2.e32};
            // complexT tmp_e22 {Mat1.e21 * Mat2.e12 + Mat1.e22 * Mat2.e22 + Mat1.e23 * Mat2.e32};
            // complexT tmp_e32 {Mat1.e31 * Mat2.e12 + Mat1.e32 * Mat2.e22 + Mat1.e33 * Mat2.e32};
            // complexT tmp_e13 {Mat1.e11 * Mat2.e13 + Mat1.e12 * Mat2.e23 + Mat1.e13 * Mat2.e33};
            // complexT tmp_e23 {Mat1.e21 * Mat2.e13 + Mat1.e22 * Mat2.e23 + Mat1.e23 * Mat2.e33};
            // complexT tmp_e33 {Mat1.e31 * Mat2.e13 + Mat1.e32 * Mat2.e23 + Mat1.e33 * Mat2.e33};
            // Mat1.e11 = tmp_e11; Mat1.e21 = tmp_e21; Mat1.e31 = tmp_e31;
            // Mat1.e12 = tmp_e12; Mat1.e22 = tmp_e22; Mat1.e32 = tmp_e32;
            // Mat1.e13 = tmp_e13; Mat1.e23 = tmp_e23; Mat1.e33 = tmp_e33;
            // Naive multiplication (TODO: try out if making a temporary copy helps with vectorization)
            // for (int i = 0; i < Mat1.Nrow; ++i)
            // {
            //     std::array<complexT, 3> tmp_row;
            //     for (int j = 0; j < Mat1.Ncolumn; ++j)
            //     {
            //         // complexT tmp {static_cast<complexT>(0.0)};
            //         // #pragma omp simd reduction(+: tmp)
            //         for (int k = 0; k < Mat2.Ncolumn; ++k)
            //         {
            //             tmp_row[j] += Mat1(i, k) * Mat2(k, j);
            //             // std::cout << Mat1(i, k) << " * " << Mat2(k, j) << ", " << tmp << std::endl;
            //             // tmp += Mat1(k, i) * Mat2(j, k);
            //         }
            //     }
            //     Mat1(i, 0) = tmp_row[0];
            //     Mat1(i, 1) = tmp_row[1];
            //     Mat1(i, 2) = tmp_row[2];
            // }

            // Since these multiplications only work on matrices of the same shape, we can always use the shape parameters of Mat1
            // for (int j = 0; j < Mat1.Ncolumn; ++j)
            // {
            //     std::array<complexT, 3> tmp_column;
            //     for (int i = 0; i < Mat1.Nrow; ++i)
            //     for (int k = 0; k < Mat1.Nrow; ++k)
            //     {
            //         tmp_column[i] += Mat1(i, k) * Mat2(k, j);
            //     }
            //     Mat1(0, j) = tmp_column[0];
            //     Mat1(1, j) = tmp_column[1];
            //     Mat1(2, j) = tmp_column[2];
            // }
            // for (int j = 0; j < Mat1.Nrow; ++j)
            // for (int k = 0; k < Mat1.Nrow; ++k)
            // for (int i = 0; i < Mat1.Nrow; ++i)
            // {
            //     Mat_tmp(i, j) += Mat1(i, k) * Mat2(k, j);
            // }
            // Matrix_3x3_ Mat_tmp;
            // Mat_tmp.data[0] = Mat1.data[0] * Mat2.data[0] + Mat1.data[3] * Mat2.data[1] + Mat1.data[6] * Mat2.data[2];
            // Mat_tmp.data[1] = Mat1.data[1] * Mat2.data[0] + Mat1.data[4] * Mat2.data[1] + Mat1.data[7] * Mat2.data[2];
            // Mat_tmp.data[2] = Mat1.data[2] * Mat2.data[0] + Mat1.data[5] * Mat2.data[1] + Mat1.data[8] * Mat2.data[2];
            // Mat_tmp.data[3] = Mat1.data[0] * Mat2.data[3] + Mat1.data[3] * Mat2.data[4] + Mat1.data[6] * Mat2.data[5];
            // Mat_tmp.data[4] = Mat1.data[1] * Mat2.data[3] + Mat1.data[4] * Mat2.data[4] + Mat1.data[7] * Mat2.data[5];
            // Mat_tmp.data[5] = Mat1.data[2] * Mat2.data[3] + Mat1.data[5] * Mat2.data[4] + Mat1.data[8] * Mat2.data[5];
            // Mat_tmp.data[6] = Mat1.data[0] * Mat2.data[6] + Mat1.data[3] * Mat2.data[7] + Mat1.data[6] * Mat2.data[8];
            // Mat_tmp.data[7] = Mat1.data[1] * Mat2.data[6] + Mat1.data[4] * Mat2.data[7] + Mat1.data[7] * Mat2.data[8];
            // Mat_tmp.data[8] = Mat1.data[2] * Mat2.data[6] + Mat1.data[5] * Mat2.data[7] + Mat1.data[8] * Mat2.data[8];
            // Mat1 = Mat_tmp;
            complexT tmp0 = Mat1.data[0] * Mat2.data[0] + Mat1.data[3] * Mat2.data[1] + Mat1.data[6] * Mat2.data[2];
            complexT tmp1 = Mat1.data[1] * Mat2.data[0] + Mat1.data[4] * Mat2.data[1] + Mat1.data[7] * Mat2.data[2];
            complexT tmp2 = Mat1.data[2] * Mat2.data[0] + Mat1.data[5] * Mat2.data[1] + Mat1.data[8] * Mat2.data[2];
            complexT tmp3 = Mat1.data[0] * Mat2.data[3] + Mat1.data[3] * Mat2.data[4] + Mat1.data[6] * Mat2.data[5];
            complexT tmp4 = Mat1.data[1] * Mat2.data[3] + Mat1.data[4] * Mat2.data[4] + Mat1.data[7] * Mat2.data[5];
            complexT tmp5 = Mat1.data[2] * Mat2.data[3] + Mat1.data[5] * Mat2.data[4] + Mat1.data[8] * Mat2.data[5];
            complexT tmp6 = Mat1.data[0] * Mat2.data[6] + Mat1.data[3] * Mat2.data[7] + Mat1.data[6] * Mat2.data[8];
            complexT tmp7 = Mat1.data[1] * Mat2.data[6] + Mat1.data[4] * Mat2.data[7] + Mat1.data[7] * Mat2.data[8];
            complexT tmp8 = Mat1.data[2] * Mat2.data[6] + Mat1.data[5] * Mat2.data[7] + Mat1.data[8] * Mat2.data[8];
            Mat1.data = {tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8};
            return Mat1;
        }

        friend Matrix_3x3_ operator+(const Matrix_3x3_& Mat1, const Matrix_3x3_& Mat2) noexcept
        {
            Matrix_3x3_ tmp {Mat1};
            return tmp += Mat2;
        }

        friend Matrix_3x3_ operator-(const Matrix_3x3_& Mat1, const Matrix_3x3_& Mat2) noexcept
        {
            Matrix_3x3_ tmp {Mat1};
            return tmp -= Mat2;
        }

        friend Matrix_3x3_ operator*(const Matrix_3x3_& Mat1, const Matrix_3x3_& Mat2) noexcept
        {
            Matrix_3x3_ tmp {Mat1};
            return tmp *= Mat2;
        }

        // // Multiplication functions (not operators)
        // friend Matrix_3x3_ mult_oo(const Matrix_3x3_& Mat1, const Matrix_3x3_& Mat2) noexcept
        // {
        //     Matrix_3x3_ tmp;
        //     // #pragma omp simd
        //     for (int i = 0; i < Mat1.Nrow;    ++i)
        //     for (int j = 0; j < Mat1.Ncolumn; ++j)
        //     {
        //         for (int k = 0; k < Mat2.Ncolumn; ++k)
        //         {
        //             tmp(i, j) += Mat1(i, k) * Mat2(k, j);
        //         }
        //     }
        //     return tmp;
        // }

        // friend Matrix_3x3_ mult_od(const Matrix_3x3_& Mat1, const Matrix_3x3_& Mat2) noexcept
        // {
        //     Matrix_3x3_ tmp;
        //     // #pragma omp simd
        //     for (int i = 0; i < Mat1.Nrow;    ++i)
        //     for (int j = 0; j < Mat1.Ncolumn; ++j)
        //     {
        //         for (int k = 0; k < Mat2.Ncolumn; ++k)
        //         {
        //             tmp(i, j) += Mat1(i, k) * std::conj(Mat2(j, k));
        //         }
        //     }
        //     return tmp;
        // }

        // friend Matrix_3x3_ mult_do(const Matrix_3x3_& Mat1, const Matrix_3x3_& Mat2) noexcept
        // {
        //     Matrix_3x3_ tmp;
        //     // #pragma omp simd
        //     for (int i = 0; i < Mat1.Nrow;    ++i)
        //     for (int j = 0; j < Mat1.Ncolumn; ++j)
        //     {
        //         for (int k = 0; k < Mat2.Ncolumn; ++k)
        //         {
        //             tmp(i, j) += std::conj(Mat1(k, i)) * Mat2(k, j);
        //         }
        //     }
        //     return tmp;
        // }

        // friend Matrix_3x3_ mult_dd(const Matrix_3x3_& Mat1, const Matrix_3x3_& Mat2) noexcept
        // {
        //     Matrix_3x3_ tmp;
        //     // #pragma omp simd
        //     for (int i = 0; i < Mat1.Nrow;    ++i)
        //     for (int j = 0; j < Mat1.Ncolumn; ++j)
        //     {
        //         for (int k = 0; k < Mat2.Ncolumn; ++k)
        //         {
        //             tmp(i, j) += Mat1(k, i) * Mat2(j, k);
        //         }
        //     }
        //     return tmp;
        // }

        // [[deprecated]]
        complexT operator()(const int i, const int j) const noexcept
        {
            // switch(Ncolumn * i + j)
            // {
            //     case 0:
            //         return e11;
            //     case 1:
            //         return e12;
            //     case 2:
            //         return e13;
            //     case 3:
            //         return e21;
            //     case 4:
            //         return e22;
            //     case 5:
            //         return e23;
            //     case 6:
            //         return e31;
            //     case 7:
            //         return e32;
            //     case 8:
            //         return e33;
            // }
            // Column-major
            // return data[Ncolumn * i + j];
            // Row-major
            return data[i + Nrow * j];
        }

        // [[deprecated]]
        complexT& operator()(const int i, const int j) noexcept
        {
            // switch(Ncolumn * i + j)
            // {
            //     case 0:
            //         return e11;
            //     case 1:
            //         return e12;
            //     case 2:
            //         return e13;
            //     case 3:
            //         return e21;
            //     case 4:
            //         return e22;
            //     case 5:
            //         return e23;
            //     case 6:
            //         return e31;
            //     case 7:
            //         return e32;
            //     case 8:
            //         return e33;
            // }
            // Column-major
            // return data[Ncolumn * i + j];
            // Row-major
            return data[i + Nrow * j];
        }

        // The determinant of a SU(3) matrix is 1, but since this class may also be used to represent general 3x3 matrices this function doesn't always return 1
        // TODO: Replace with LU decomposition?
        complexT determinant() const noexcept
        {
            // return e11 * (e22 * e33 - e23 * e32)
            //      - e12 * (e21 * e33 - e23 * e31)
            //      + e13 * (e21 * e32 - e22 * e31);
            return data[0] * (data[4] * data[8] - data[7] * data[5])
                 - data[3] * (data[1] * data[8] - data[7] * data[2])
                 + data[6] * (data[1] * data[5] - data[4] * data[2]);
            // data({e11_in, e21_in, e31_in, e12_in, e22_in, e32_in, e13_in, e23_in, e33_in})
        }

        complexT trace() const noexcept
        {
            // return e11 + e22 + e33;
            return data[0] + data[4] + data[8];
        }

        floatT norm() const noexcept
        {
            // Old
            // return std::sqrt(std::abs(e11) * std::abs(e11) + std::abs(e12) * std::abs(e12) + std::abs(e13) * std::abs(e13)
            //                + std::abs(e21) * std::abs(e21) + std::abs(e22) * std::abs(e22) + std::abs(e23) * std::abs(e23)
            //                + std::abs(e31) * std::abs(e31) + std::abs(e32) * std::abs(e32) + std::abs(e33) * std::abs(e33));
            return std::sqrt(std::abs(data[0]) * std::abs(data[0]) + std::abs(data[1]) * std::abs(data[1]) + std::abs(data[2]) * std::abs(data[2])
                           + std::abs(data[3]) * std::abs(data[3]) + std::abs(data[4]) * std::abs(data[4]) + std::abs(data[5]) * std::abs(data[5])
                           + std::abs(data[6]) * std::abs(data[6]) + std::abs(data[7]) * std::abs(data[7]) + std::abs(data[8]) * std::abs(data[8]));
            // floatT norm_tmp {0.0};
            // for (int i = 0; i < Nelements; ++i)
            // {
            //     norm_tmp += std::abs(data[i]) * std::abs(data[i]);
            // }
            // return std::sqrt(norm_tmp);
            // New
            // return std::sqrt(std::abs(e11) * std::abs(e11) + std::abs(e21) * std::abs(e21) + std::abs(e31) * std::abs(e31)
            //                + std::abs(e12) * std::abs(e12) + std::abs(e22) * std::abs(e22) + std::abs(e32) * std::abs(e32)
            //                + std::abs(e13) * std::abs(e13) + std::abs(e23) * std::abs(e23) + std::abs(e33) * std::abs(e33));
        }

        Matrix_3x3_ adjoint() const noexcept
        {
            // return {std::conj(e11), std::conj(e21), std::conj(e31),
            //         std::conj(e12), std::conj(e22), std::conj(e32),
            //         std::conj(e13), std::conj(e23), std::conj(e33)};
            return {std::conj(data[0]), std::conj(data[1]), std::conj(data[2]),
                    std::conj(data[3]), std::conj(data[4]), std::conj(data[5]),
                    std::conj(data[6]), std::conj(data[7]), std::conj(data[8])};
        }

        Matrix_3x3_ setZero() noexcept
        {
            // Old
            // e11 = static_cast<complexT>(0.0); e12 = static_cast<complexT>(0.0); e13 = static_cast<complexT>(0.0);
            // e21 = static_cast<complexT>(0.0); e22 = static_cast<complexT>(0.0); e23 = static_cast<complexT>(0.0);
            // e31 = static_cast<complexT>(0.0); e32 = static_cast<complexT>(0.0); e33 = static_cast<complexT>(0.0);
            // New
            // e11 = static_cast<complexT>(0.0); e21 = static_cast<complexT>(0.0); e31 = static_cast<complexT>(0.0);
            // e12 = static_cast<complexT>(0.0); e22 = static_cast<complexT>(0.0); e32 = static_cast<complexT>(0.0);
            // e13 = static_cast<complexT>(0.0); e23 = static_cast<complexT>(0.0); e33 = static_cast<complexT>(0.0);
            for (int i = 0; i < Nelements; ++i)
            {
                data[i] = static_cast<complexT>(0.0);
            }
            return *this;
        }

        Matrix_3x3_ setIdentity() noexcept
        {
            // Old
            // e11 = static_cast<complexT>(1.0); e12 = static_cast<complexT>(0.0); e13 = static_cast<complexT>(0.0);
            // e21 = static_cast<complexT>(0.0); e22 = static_cast<complexT>(1.0); e23 = static_cast<complexT>(0.0);
            // e31 = static_cast<complexT>(0.0); e32 = static_cast<complexT>(0.0); e33 = static_cast<complexT>(1.0);
            // New
            // e11 = static_cast<complexT>(1.0); e21 = static_cast<complexT>(0.0); e31 = static_cast<complexT>(0.0);
            // e12 = static_cast<complexT>(0.0); e22 = static_cast<complexT>(1.0); e32 = static_cast<complexT>(0.0);
            // e13 = static_cast<complexT>(0.0); e23 = static_cast<complexT>(0.0); e33 = static_cast<complexT>(1.0);
            // for (int i = 0; i < Nelements; ++i)
            // {
            //     data[i] = static_cast<complexT>(1.0);
            // }
            data = Matrix_3x3_::Identity().data;
            return *this;
        }

        static constexpr Matrix_3x3_ Zero() noexcept
        {
            return {static_cast<complexT>(0.0), static_cast<complexT>(0.0), static_cast<complexT>(0.0),
                    static_cast<complexT>(0.0), static_cast<complexT>(0.0), static_cast<complexT>(0.0),
                    static_cast<complexT>(0.0), static_cast<complexT>(0.0), static_cast<complexT>(0.0)};
        }

        static constexpr Matrix_3x3_ Identity() noexcept
        {
            // Matrix_3x3_ tmp {static_cast<complexT>(1.0), static_cast<complexT>(0.0), static_cast<complexT>(0.0),
            //                  static_cast<complexT>(0.0), static_cast<complexT>(1.0), static_cast<complexT>(0.0),
            //                  static_cast<complexT>(0.0), static_cast<complexT>(0.0), static_cast<complexT>(1.0)};
            // return tmp;
            return {static_cast<complexT>(1.0), static_cast<complexT>(0.0), static_cast<complexT>(0.0),
                    static_cast<complexT>(0.0), static_cast<complexT>(1.0), static_cast<complexT>(0.0),
                    static_cast<complexT>(0.0), static_cast<complexT>(0.0), static_cast<complexT>(1.0)};
        }

        friend std::ostream& operator<<(std::ostream& stream, const Matrix_3x3_& Mat)
        {
            // TODO: Separate with commas or not?
            // return stream << Mat.e11 << ", " << Mat.e12 << ", " << Mat.e13 << "\n"
            //               << Mat.e21 << ", " << Mat.e22 << ", " << Mat.e23 << "\n"
            //               << Mat.e31 << ", " << Mat.e32 << ", " << Mat.e33;
            return stream << Mat.data[0] << ", " << Mat.data[3] << ", " << Mat.data[6] << "\n"
                          << Mat.data[1] << ", " << Mat.data[4] << ", " << Mat.data[7] << "\n"
                          << Mat.data[2] << ", " << Mat.data[5] << ", " << Mat.data[8];
        }
    private:
        //...
};

#endif // LETTUCE_CUSTOM_SU3_HPP
