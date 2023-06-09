#ifndef LETTUCE_DEFINES_HPP
#define LETTUCE_DEFINES_HPP

// Non-standard library headers
#include "../PCG/pcg_random.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
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

inline constexpr int Nt {24};
inline constexpr int Nx {24};
inline constexpr int Ny {24};
inline constexpr int Nz {24};

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
inline constexpr int n_heatbath {1};                        // Number of heat bath sweeps per total update sweep
inline constexpr int n_hmc {0};                             // Number of integration steps per HMC update
inline constexpr int n_orelax {4};                          // Number of overrelaxation sweeps per total update sweep
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
uint_fast64_t acceptance_count                   {0};       // Metropolis acceptance rate for new configurations
uint_fast64_t acceptance_count_or                {0};       // Overrelaxation acceptance rate
uint_fast64_t acceptance_count_hmc               {0};       // HMC acceptance rate
uint_fast64_t acceptance_count_metadynamics_hmc  {0};       // MetaD-HMC acceptance rate
uint_fast64_t acceptance_count_tempering         {0};       // Parallel tempering swap acceptance rate
uint_fast64_t acceptance_count_instanton         {0};       // Instanton update acceptance rate

//-----
// NxN_matrix is the same type as SUN_matrix, the different names are only meant to distinguish between
// SU(N) group elements and NxN matrices mathematically
using Matrix_2x2     = Eigen::Matrix<std::complex<floatT>, 2, 2>;
using Matrix_SU2     = Matrix_2x2;
using Matrix_3x3     = Eigen::Matrix<std::complex<floatT>, 3, 3>;
using Matrix_SU3     = Matrix_3x3;
using Local_tensor   = std::array<std::array<Matrix_SU3, 4>, 4>;

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
