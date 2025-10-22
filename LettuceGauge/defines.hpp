#ifndef LETTUCE_DEFINES_HPP
#define LETTUCE_DEFINES_HPP

// Non-standard library headers
#include "../PCG/pcg_random.hpp"
#include "random/prng_wrapper.hpp"
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
#include <concepts>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
//----------------------------------------
// Standard C headers
// ...

std::string program_version = "SU(3)_version_1.3";

inline constexpr int Ndim   {4};
inline constexpr int Ncolor {3};
static_assert(Ndim   == 4, "Currently only 4 dimensions are supported!");
static_assert(Ncolor == 3, "Currently only SU(3) is supported!");

inline constexpr int Nt {24};
inline constexpr int Nx {24};
inline constexpr int Ny {24};
inline constexpr int Nz {24};

template<typename T>
inline constexpr std::complex<T> i(0, 1);
template<typename T>
inline constexpr T pi {static_cast<T>(3.14159265358979323846L)};

using floatT = double;

// Various global parameters (yes this is ugly)

// Run parameters
int n_run;                                                  // Number of runs
int expectation_period;                                     // Number of updates between calculation of expectation values
floatT beta;                                                // Coupling
inline int n_smear {10};                                    // Number of smearing steps (total amount of smearing steps is actually n_smear * n_smear_skip)
inline int n_smear_skip {10};                               // Number of smearing steps to skip between measurements
inline floatT rho_stout {0.10};                             // Stout smearing parameter
inline floatT rho_stout_metadynamics {0.12};                // Stout smearing parameter for Metadynamics CV
// Checkpointing
int checkpoint_period {500};                                // Number of updates between checkpoints (both for the configuration and the PRNG state)
int n_checkpoint_backups {2};                               // Number of rotating checkpoints to use
inline bool extend_run {false};                             // Set to true only if extending an existing run (among other things skips thermalization)
// Update algorithm related parameters
inline int n_therm {20};                                     // Number of update sweeps before starting actual update loop (the type of update sweeps is specified below)
inline constexpr int n_metro {0};                           // Number of Metropolis sweeps per total update sweep
inline constexpr int multi_hit {8};                         // Number of hits per site in Metropolis algorithm
inline constexpr int n_heatbath {0};                        // Number of heat bath sweeps per total update sweep
inline constexpr int n_hmc {5};                             // Number of integration steps per HMC update
inline constexpr double hmc_trajectory_length {1.0};        // Trajectory length of a single HMC update
inline constexpr int n_orelax {0};                          // Number of overrelaxation sweeps per total update sweep
inline constexpr int n_instanton_update {0};                // Number of instanton updates per total update sweep
inline constexpr bool metadynamics_enabled {true};          // Enable metadynamics updates or not
inline constexpr int  metapotential_update_stride {1};      // An update stride of 0 is interpreted as a static bias potential
inline constexpr bool metapotential_well_tempered {false};   // If true, use well tempered Metadynamics instead of standard Metadynamics updates
inline constexpr bool metadynamics_path_update_enabled {true};
inline constexpr int n_smear_meta {4};                      // Number of smearing steps for topological charge used in Metadynamics
inline constexpr bool tempering_enabled {false};             // Enable metadynamics updates with tempering or not
inline constexpr int tempering_nonmetadynamics_sweeps {10};  // Number of non metadynamics update sweeps for every metadynamics update during tempering
inline constexpr int tempering_swap_period {1};             // Number of update sweeps between parallel tempering swap attempts
inline double metro_target_acceptance {0.5};                // Target acceptance rate for Metropolis update, values around 50% seem to work well, but TRY OUT!
double DeltaH;                                              // Energy change during HMC trajectory (declared globally so we can print it independently as observable)
double DeltaVTempering;                                     // Metapotential change of tempering swap proposal
double DeltaSInstanton;                                     // Action change of instanton update proposal (see above)
double JacobianInstanton;                                   // Jacobian during instanton update with gradient flow
// Directory and logfile paths
std::string old_maindirectory;                              // Main directory of previous run we wish to extend
std::string maindirectory;                                  // Main directory containing all other subdirectories and files
std::string checkpointdirectory;                            // Default directory to save checkpoints to (configs and PRNG states)
std::string logfilepath;                                    // Filepath (log)
std::string parameterfilepath;                              // Filepath (parameters)
std::string hmclogfilepath;                                 // Filepath (HMC log)
std::string metapotentialfilepath;                          // Filepath (metapotential)
std::string logfilepath_temper;                             // Filepath (log for run with bias potential during PT-MetaD runs)
auto start {std::chrono::system_clock::now()};              // Start time
std::ofstream datalog;                                      // Output stream to save data
std::ofstream wilsonlog;                                    // Output stream to save data (Wilson loops)
// PRNG stuff
pcg_extras::seed_seq_from<std::random_device> seed_source;  // Seed source to seed PRNGs
#ifdef FIXED_SEED                                           // PRNG for random coordinates and probabilites
pcg64 generator_rand(1);
#else
pcg64 generator_rand(seed_source);
#endif
PRNG4D<Nt, Nx, Ny, Nz, pcg64, floatT, int>    global_prng(generator_rand);
// For tracking various acceptance rates of update algorithms
uint_fast64_t acceptance_count                   {0};       // Metropolis acceptance rate for new configurations
uint_fast64_t acceptance_count_or                {0};       // Overrelaxation acceptance rate
uint_fast64_t acceptance_count_hmc               {0};       // HMC acceptance rate
uint_fast64_t acceptance_count_metadynamics_hmc  {0};       // MetaD-HMC acceptance rate
uint_fast64_t acceptance_count_tempering         {0};       // Parallel tempering swap acceptance rate
uint_fast64_t acceptance_count_instanton         {0};       // Instanton update acceptance rate

// Matrix_NxN is the same type as MatrixSUN, but the different names may be useful to explicitly indicate which objects are group elements and which are not
using Matrix_2x2     = Eigen::Matrix<std::complex<floatT>, 2, 2, Eigen::RowMajor>;
using Matrix_SU2     = Matrix_2x2;
using Matrix_3x3     = Eigen::Matrix<std::complex<floatT>, 3, 3, Eigen::RowMajor>;
using Matrix_SU3     = Matrix_3x3;
using Local_tensor   = std::array<std::array<Matrix_SU3, 4>, 4>;

// Trick to allow type promotion below (for less annoying complex numbers)
template<typename T>
struct identity_t
{
    using type = T;
};

// Make working with std::complex<> numbers suck less... allow promotion.
#define COMPLEX_OPS(OP)                                                                \
  template<typename _Tp>                                                               \
  std::complex<_Tp>                                                                    \
  operator OP(std::complex<_Tp> lhs, const typename identity_t<_Tp>::type& rhs)        \
  {                                                                                    \
    return lhs OP rhs;                                                                 \
  }                                                                                    \
  template<typename _Tp>                                                               \
  std::complex<_Tp>                                                                    \
  operator OP(const typename identity_t<_Tp>::type& lhs, const std::complex<_Tp>& rhs) \
  {                                                                                    \
    return lhs OP rhs;                                                                 \
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
