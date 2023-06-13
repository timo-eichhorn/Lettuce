#ifndef LETTUCE_DEFINES_HPP
#define LETTUCE_DEFINES_HPP

// Non-standard library headers
#include "../PCG/pcg_random.hpp"
#include "random/prng_wrapper.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <array>
#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
//----------------------------------------
// Standard C headers
// ...

//-----

std::string program_version = "SU(3)_version_1.3";

//-----

inline constexpr int Nt {12};
inline constexpr int Nx {12};
inline constexpr int Ny {12};
inline constexpr int Nz {12};

template<typename T>
inline constexpr std::complex<T> i(0, 1);
template<typename T>
inline constexpr T pi {static_cast<T>(3.14159265358979323846L)};

using floatT = double;

//-----

int n_run;                                                  // Number of runs
double n_run_inverse;                                       // Inverse number of runs
int expectation_period;                                     // Number of updates between calculation of expectation values
inline int n_smear {7};                                     // Number of smearing steps (total amount of smearing steps is actually n_smear * n_smear_skip)
inline int n_smear_skip {5};                                // Number of smearing steps to skip between measurements
inline floatT rho_stout {0.12};                             // Stout smearing parameter
inline constexpr int n_metro {0};                           // Number of Metropolis sweeps per total update sweep
inline constexpr int multi_hit {8};                         // Number of hits per site in Metropolis algorithm
inline constexpr int n_heatbath {0};                        // Number of heat bath sweeps per total update sweep
inline constexpr int n_hmc {5};                             // Number of integration steps per HMC update
inline constexpr int n_orelax {0};                          // Number of overrelaxation sweeps per total update sweep
inline constexpr int n_instanton_update {0};                // Number of instanton updates per total update sweep
inline constexpr bool metadynamics_enabled {false};          // Enable metadynamics updates or not
inline constexpr bool metapotential_updated {false};         // If true, update the metapotential with every update, if false, simulate with a static metapotential
inline constexpr int n_smear_meta {4};                      // Number of smearing steps for topological charge used in Metadynamics
inline constexpr bool tempering_enabled {false};             // Enable metadynamics updates with tempering or not
inline constexpr int tempering_nonmetadynamics_sweeps {10};  // Number of non metadynamics update sweeps for every metadynamics update during tempering
inline constexpr int tempering_swap_period {1};             // Number of update sweeps between parallel tempering swap attempts
inline double metro_norm {1.0};                             // Norm for Metropolis update. CAUTION: Needs to be set to correct value inside Configuration() function
inline double metro_target_acceptance {0.5};                // Target acceptance rate for Metropolis update, values around 50% seem to work well, but TRY OUT!
double DeltaH;                                              // Energy change during HMC trajectory (declared globally so we can print it independently as observable)
double DeltaVTempering;                                     // Metapotential change of tempering swap proposal
double DeltaSInstanton;                                     // Action change of instanton update proposal (see above)
double JacobianInstanton;                                   // Jacobian during instanton update with gradient flow
std::string directoryname;                                  // Directory name
std::string logfilepath;                                    // Filepath (log)
std::string parameterfilepath;                              // Filepath (parameters)
std::string wilsonfilepath;                                 // Filepath (Wilson loops)
std::string metapotentialfilepath;                          // Filepath (metapotential)
auto start {std::chrono::system_clock::now()};              // Start time
floatT beta;                                                // Coupling
std::ofstream datalog;                                      // Output stream to save data
std::ofstream wilsonlog;                                    // Output stream to save data (Wilson loops)
pcg_extras::seed_seq_from<std::random_device> seed_source;  // Seed source to seed PRNGs
#ifdef FIXED_SEED                                           // PRNG for random coordinates and probabilites
pcg64 generator_rand(1);
#else
pcg64 generator_rand(seed_source);
#endif
std::vector<pcg64> prng_vector;                             // Vector of PRNGs for parallel usage
std::vector<std::normal_distribution<floatT>> ndist_vector; // Vector of normal distributions for parallel usage in HMC
// PRNG<pcg64, floatT, int> global_prng((Nt * Nx * Ny * Nz), seed_source);
uint_fast64_t acceptance_count                   {0};       // Metropolis acceptance rate for new configurations
uint_fast64_t acceptance_count_or                {0};       // Overrelaxation acceptance rate
uint_fast64_t acceptance_count_hmc               {0};       // HMC acceptance rate
uint_fast64_t acceptance_count_metadynamics_hmc  {0};       // MetaD-HMC acceptance rate
uint_fast64_t acceptance_count_tempering         {0};       // Parallel tempering swap acceptance rate
uint_fast64_t acceptance_count_instanton         {0};       // Instanton update acceptance rate

//-----

template<typename floatT>
class Matrix_3x3_
{
    using complexT = std::complex<floatT>;
    public:
        // "A static data member is not part of the subobjects of a class" [class.static.data]
        static constexpr int Nrow    {3};
        static constexpr int Ncolumn {3};
        static constexpr int Ncolor  {3};
        // Matrix_3x3_ identity() {
        //     Matrix_3x3_ tmp;
        //     tmp[0]
        // } 
        // std::array<complexT, Ncol> Mat;
        // Standard C++ order/row-major:
        // complexT e11, e12, e13, e21, e22, e23, e31, e32, e33;
        // Eigen order/column-major:
        complexT e11, e21, e31, e12, e22, e32, e13, e23, e33;


        Matrix_3x3_() noexcept = default;

        // TODO: In which order should the constructor accept the arguments? Memory layout, or layout on paper?
        Matrix_3x3_(const complexT e11_in, const complexT e12_in, const complexT e13_in,
            const complexT e21_in, const complexT e22_in, const complexT e23_in,
            const complexT e31_in, const complexT e32_in, const complexT e33_in) noexcept :
        // Note that the members are initialized in the same order as they are declared above, not in the order that they appear in the initializer list below
        e11(e11_in), e12(e12_in), e13(e13_in),
        e21(e21_in), e22(e22_in), e23(e23_in),
        e31(e31_in), e32(e32_in), e33(e33_in)
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
            e11 = Mat.e11; e21 = Mat.e21; e31 = Mat.e31;
            e12 = Mat.e12; e22 = Mat.e22; e32 = Mat.e32;
            e13 = Mat.e13; e23 = Mat.e23; e33 = Mat.e33;
            return *this;
        }

        // Delete initializer list copy constructor
        template<typename T>
        Matrix_3x3_& operator=(std::initializer_list<T>) = delete;

        // TODO: Problematic, compiler complains about ambiguity between unary and binary operator -
        friend Matrix_3x3_ operator-(const Matrix_3x3_& Mat) noexcept
        {
            return {-Mat.e11, -Mat.e12, -Mat.e13,
                    -Mat.e21, -Mat.e22, -Mat.e23,
                    -Mat.e31, -Mat.e32, -Mat.e33,};
        }

        friend Matrix_3x3_& operator+=(Matrix_3x3_& Mat1, const Matrix_3x3_& Mat2) noexcept
        {
            // Old
            // Mat1.e11 += Mat2.e11; Mat1.e12 += Mat2.e12; Mat1.e13 += Mat2.e13;
            // Mat1.e21 += Mat2.e21; Mat1.e22 += Mat2.e22; Mat1.e23 += Mat2.e23;
            // Mat1.e31 += Mat2.e31; Mat1.e32 += Mat2.e32; Mat1.e33 += Mat2.e33;
            // New
            Mat1.e11 += Mat2.e11; Mat1.e21 += Mat2.e21; Mat1.e31 += Mat2.e31;
            Mat1.e12 += Mat2.e12; Mat1.e22 += Mat2.e22; Mat1.e32 += Mat2.e32;
            Mat1.e13 += Mat2.e13; Mat1.e23 += Mat2.e23; Mat1.e33 += Mat2.e33;
            return Mat1;
        }

        friend Matrix_3x3_& operator-=(Matrix_3x3_& Mat1, const Matrix_3x3_& Mat2) noexcept
        {
            // Old
            // Mat1.e11 -= Mat2.e11; Mat1.e12 -= Mat2.e12; Mat1.e13 -= Mat2.e13;
            // Mat1.e21 -= Mat2.e21; Mat1.e22 -= Mat2.e22; Mat1.e23 -= Mat2.e23;
            // Mat1.e31 -= Mat2.e31; Mat1.e32 -= Mat2.e32; Mat1.e33 -= Mat2.e33;
            // New
            Mat1.e11 -= Mat2.e11; Mat1.e21 -= Mat2.e21; Mat1.e31 -= Mat2.e31;
            Mat1.e12 -= Mat2.e12; Mat1.e22 -= Mat2.e22; Mat1.e32 -= Mat2.e32;
            Mat1.e13 -= Mat2.e13; Mat1.e23 -= Mat2.e23; Mat1.e33 -= Mat2.e33;
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
            Mat.e11 *= a; Mat.e21 *= a; Mat.e31 *= a;
            Mat.e12 *= a; Mat.e22 *= a; Mat.e32 *= a;
            Mat.e13 *= a; Mat.e23 *= a; Mat.e33 *= a;
            return Mat;
        }

        friend Matrix_3x3_& operator*=(Matrix_3x3_& Mat, const complexT a) noexcept
        {
            // Old
            // Mat.e11 *= a; Mat.e12 *= a; Mat.e13 *= a;
            // Mat.e21 *= a; Mat.e22 *= a; Mat.e23 *= a;
            // Mat.e31 *= a; Mat.e32 *= a; Mat.e33 *= a;
            // New
            Mat.e11 *= a; Mat.e21 *= a; Mat.e31 *= a;
            Mat.e12 *= a; Mat.e22 *= a; Mat.e32 *= a;
            Mat.e13 *= a; Mat.e23 *= a; Mat.e33 *= a;
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
            complexT tmp_e11 {Mat1.e11 * Mat2.e11 + Mat1.e12 * Mat2.e21 + Mat1.e13 * Mat2.e31};
            complexT tmp_e21 {Mat1.e21 * Mat2.e11 + Mat1.e22 * Mat2.e21 + Mat1.e23 * Mat2.e31};
            complexT tmp_e31 {Mat1.e31 * Mat2.e11 + Mat1.e32 * Mat2.e21 + Mat1.e33 * Mat2.e31};
            complexT tmp_e12 {Mat1.e11 * Mat2.e12 + Mat1.e12 * Mat2.e22 + Mat1.e13 * Mat2.e32};
            complexT tmp_e22 {Mat1.e21 * Mat2.e12 + Mat1.e22 * Mat2.e22 + Mat1.e23 * Mat2.e32};
            complexT tmp_e32 {Mat1.e31 * Mat2.e12 + Mat1.e32 * Mat2.e22 + Mat1.e33 * Mat2.e32};
            complexT tmp_e13 {Mat1.e11 * Mat2.e13 + Mat1.e12 * Mat2.e23 + Mat1.e13 * Mat2.e33};
            complexT tmp_e23 {Mat1.e21 * Mat2.e13 + Mat1.e22 * Mat2.e23 + Mat1.e23 * Mat2.e33};
            complexT tmp_e33 {Mat1.e31 * Mat2.e13 + Mat1.e32 * Mat2.e23 + Mat1.e33 * Mat2.e33};
            Mat1.e11 = tmp_e11; Mat1.e21 = tmp_e21; Mat1.e31 = tmp_e31;
            Mat1.e12 = tmp_e12; Mat1.e22 = tmp_e22; Mat1.e32 = tmp_e32;
            Mat1.e13 = tmp_e13; Mat1.e23 = tmp_e23; Mat1.e33 = tmp_e33;
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

        complexT operator()(const int i, const int j) const noexcept
        {
            switch(Ncolumn * i + j)
            {
                case 0:
                    return e11;
                case 1:
                    return e12;
                case 2:
                    return e13;
                case 3:
                    return e21;
                case 4:
                    return e22;
                case 5:
                    return e23;
                case 6:
                    return e31;
                case 7:
                    return e32;
                case 8:
                    return e33;
            }
        }

        complexT& operator()(const int i, const int j) noexcept
        {
            switch(Ncolumn * i + j)
            {
                case 0:
                    return e11;
                case 1:
                    return e12;
                case 2:
                    return e13;
                case 3:
                    return e21;
                case 4:
                    return e22;
                case 5:
                    return e23;
                case 6:
                    return e31;
                case 7:
                    return e32;
                case 8:
                    return e33;
            }
        }

        // The determinant of a SU(3) matrix is 1, but since this class may also be used to represent general 3x3 matrices this function doesn't always return 1
        // TODO: Replace with LU decomposition?
        complexT determinant() const noexcept
        {
            return e11 * (e22 * e33 - e23 * e32)
                 - e12 * (e21 * e33 - e23 * e31)
                 + e13 * (e21 * e32 - e22 * e31);
        }

        complexT trace() const noexcept
        {
            return e11 + e22 + e33;
        }

        floatT norm() const noexcept
        {
            // Old
            return std::sqrt(std::abs(e11) * std::abs(e11) + std::abs(e12) * std::abs(e12) + std::abs(e13) * std::abs(e13)
                           + std::abs(e21) * std::abs(e21) + std::abs(e22) * std::abs(e22) + std::abs(e23) * std::abs(e23)
                           + std::abs(e31) * std::abs(e31) + std::abs(e32) * std::abs(e32) + std::abs(e33) * std::abs(e33));
            // New
            // return std::sqrt(std::abs(e11) * std::abs(e11) + std::abs(e21) * std::abs(e21) + std::abs(e31) * std::abs(e31)
            //                + std::abs(e12) * std::abs(e12) + std::abs(e22) * std::abs(e22) + std::abs(e32) * std::abs(e32)
            //                + std::abs(e13) * std::abs(e13) + std::abs(e23) * std::abs(e23) + std::abs(e33) * std::abs(e33));
        }

        Matrix_3x3_ adjoint() const noexcept
        {
            return {std::conj(e11), std::conj(e21), std::conj(e31),
                    std::conj(e12), std::conj(e22), std::conj(e32),
                    std::conj(e13), std::conj(e23), std::conj(e33)};
        }

        Matrix_3x3_ setZero() noexcept
        {
            // Old
            // e11 = static_cast<complexT>(0.0); e12 = static_cast<complexT>(0.0); e13 = static_cast<complexT>(0.0);
            // e21 = static_cast<complexT>(0.0); e22 = static_cast<complexT>(0.0); e23 = static_cast<complexT>(0.0);
            // e31 = static_cast<complexT>(0.0); e32 = static_cast<complexT>(0.0); e33 = static_cast<complexT>(0.0);
            // New
            e11 = static_cast<complexT>(0.0); e21 = static_cast<complexT>(0.0); e31 = static_cast<complexT>(0.0);
            e12 = static_cast<complexT>(0.0); e22 = static_cast<complexT>(0.0); e32 = static_cast<complexT>(0.0);
            e13 = static_cast<complexT>(0.0); e23 = static_cast<complexT>(0.0); e33 = static_cast<complexT>(0.0);
            return *this;
        }

        Matrix_3x3_ setIdentity() noexcept
        {
            // Old
            // e11 = static_cast<complexT>(1.0); e12 = static_cast<complexT>(0.0); e13 = static_cast<complexT>(0.0);
            // e21 = static_cast<complexT>(0.0); e22 = static_cast<complexT>(1.0); e23 = static_cast<complexT>(0.0);
            // e31 = static_cast<complexT>(0.0); e32 = static_cast<complexT>(0.0); e33 = static_cast<complexT>(1.0);
            // New
            e11 = static_cast<complexT>(1.0); e21 = static_cast<complexT>(0.0); e31 = static_cast<complexT>(0.0);
            e12 = static_cast<complexT>(0.0); e22 = static_cast<complexT>(1.0); e32 = static_cast<complexT>(0.0);
            e13 = static_cast<complexT>(0.0); e23 = static_cast<complexT>(0.0); e33 = static_cast<complexT>(1.0);
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
            return stream << Mat.e11 << ", " << Mat.e12 << ", " << Mat.e13 << "\n"
                          << Mat.e21 << ", " << Mat.e22 << ", " << Mat.e23 << "\n"
                          << Mat.e31 << ", " << Mat.e32 << ", " << Mat.e33;
        }
    private:
        //...
};

// NxN_matrix is the same type as SUN_matrix, the different names are only meant to distinguish between
// SU(N) group elements and NxN matrices mathematically
// using Matrix_2x2     = Eigen::Matrix<std::complex<floatT>, 2, 2>;
// using Matrix_SU2     = Matrix_2x2;
// using Matrix_3x3     = Eigen::Matrix<std::complex<floatT>, 3, 3>;
// using Matrix_SU3     = Matrix_3x3;
// using Local_tensor   = std::array<std::array<Matrix_SU3, 4>, 4>;
using Matrix_3x3   = Matrix_3x3_<floatT>;
using Matrix_SU3   = Matrix_3x3;
using Local_tensor = std::array<std::array<Matrix_SU3, 4>, 4>;

//-----
// Better complex numbers?

// Trick to allow type promotion below
template<typename T>
struct identity_t { typedef T type; };

// Make working with std::complex<> numbers suck less... allow promotion.
#define COMPLEX_OPS(OP)                                                 \
  template<typename _Tp>                                               \
  std::complex<_Tp>                                                     \
  operator OP(std::complex<_Tp> lhs, const typename identity_t<_Tp>::type & rhs) \
  {                                                                     \
    return lhs OP rhs;                                                  \
  }                                                                     \
  template<typename _Tp>                                               \
  std::complex<_Tp>                                                     \
  operator OP(const typename identity_t<_Tp>::type & lhs, const std::complex<_Tp> & rhs) \
  {                                                                     \
    return lhs OP rhs;                                                  \
  }
COMPLEX_OPS(+)
COMPLEX_OPS(-)
COMPLEX_OPS(*)
COMPLEX_OPS(/)
#undef COMPLEX_OPS

// For some reason there is no inbuilt sign function???

template<typename T>
[[nodiscard]]
constexpr int sign(T x)
{
    return (x > static_cast<T>(0)) - (x < static_cast<T>(0));
}

#endif // LETTUCE_DEFINES_HPP
