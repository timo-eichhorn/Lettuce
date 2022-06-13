#ifndef LETTUCE_GLOBALS
#define LETTUCE_GLOBALS

// Non-standard library headers
#include "random/pcg/pcg_random.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
//----------------------------------------
// Standard C headers
// ...

inline std::string program_version {"SU(3)_version_1.2_metro"};

// Precision of most link variables
using float_default = float;
// Precision of accumulation variables
using float_precise = double;

//----------------------------------------
// Lattice dimensions
inline constexpr int Nt {12};
inline constexpr int Nx {12};
inline constexpr int Ny {12};
inline constexpr int Nz {12};
//----------------------------------------
// Simulation parameters known at run-time
inline float_default beta;                                  // Coupling constant
inline int n_run;                                           // Number of update sweeps
inline int expectation_period;                              // Number of update sweeps between calculation of expectation values
//----------------------------------------
// Simulation parameters known at compile-time
inline constexpr int multi_hit {8};                         // Number of hits per site in Metropolis algorithm
inline constexpr int n_smear {21};                          // Number of smearing steps
//----------------------------------------
// Constants and variables derived from simulation parameters known at compile-time
inline constexpr int volume {Nt * Nx * Ny * Nz};
inline constexpr float_precise inverse_volume {1.0 / volume};
inline constexpr float_precise update_norm {1.0 / (4.0 * volume * multi_hit)};
inline constexpr float_precise spatial_norm {1.0 / (Nx * Ny * Nz)};
//----------------------------------------
// Constants and variables derived from simulation parameters known at run-time
inline float_precise n_run_inverse;                         // Inverse number of update sweeps
//----------------------------------------
// Variables used during creation of files and folders (mostly filepaths)
auto start = std::chrono::system_clock::now();              // Start time
std::string logfilepath;                                    // Filepath (general logging of observables)
std::string parameterfilepath;                              // Filepath (separate logging of parameters)
//----------------------------------------
// Filestreams for logging data
std::ofstream datalog;                                      // Output stream to save data
//----------------------------------------
// PRNG stuff
// TODO: Is this necessary, and does this work for our random vector?
pcg_extras::seed_seq_from<std::random_device> seed_source;  // Seed source to seed PRNGs
#ifdef FIXED_SEED                                           // PRNG for random coordinates and probabilites
pcg64 generator_rand(1);
#else
pcg64 generator_rand(seed_source);
#endif
std::vector<pcg64> prng_vector;                             // Vector of PRNGs for parallel usage
uint_fast64_t acceptance_rate;                              // Acceptance rate for new configurations
uint_fast64_t acceptance_rate_or;

#endif
