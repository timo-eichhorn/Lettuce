#ifndef LETTUCE_DRESSING_HPP
#define LETTUCE_DRESSING_HPP

// Non-standard library headers
#include "ansi_colors.hpp"
#include "pcg_random.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <array>
#include <chrono>
#include <complex>
#include <fstream>
#include <random>
#include <string>
#include <vector>
// #include <utility>
//----------------------------------------
// Standard C headers
// ...

//----------------------------------------
// TODO: This file is currently never included and also has the same name as a directory
//       We should rename this file or maybe even delete it and move the components somewhere else

template<typename T>
inline constexpr std::complex<T> i(0, 1);
template<typename T>
inline constexpr T pi = static_cast<T>(3.14159265358979323846L);

//----------------------------------------
// Trick to allow type promotion below
template<typename T>
struct identity_t { typedef T type; };

// Make working with std::complex<> numbers suck less... allow promotion.
#define COMPLEX_OPS(OP)                                                 \
  template<typename _Tp>                                                \
  std::complex<_Tp>                                                     \
  operator OP(std::complex<_Tp> lhs, const typename identity_t<_Tp>::type & rhs) \
  {                                                                     \
    return lhs OP rhs;                                                  \
  }                                                                     \
  template<typename _Tp>                                                \
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

//----------------------------------------

template<typename T, size_t Nx, size_t Nt>
using GaugeField = std::array<std::array<std::array<T, 2>, Nx>, Nt>;

//----------------------------------------
// Define lattices
GaugeField<double, Nx, Nt> FineLattice;
GaugeField<double, Nx, Nt> FineLatticeCopy;
GaugeField<double, Nx/2, Nt/2> CoarseLattice;
// Define lattices for smearing
GaugeField<double, Nx, Nt> FineLatticeSmearedA;
GaugeField<double, Nx, Nt> FineLatticeSmearedB;
GaugeField<double, Nx/2, Nt/2> CoarseLatticeSmearedA;
GaugeField<double, Nx/2, Nt/2> CoarseLatticeSmearedB;
// Define lattice for Wilson flow
GaugeField<double, Nx, Nt> FineLatticeFlowed;

//----------------------------------------
// Struct containing information about the update scheme

struct UpdateScheme
{
    bool multiscale_enabled;
    bool metadynamics_enabled;
};

//----------------------------------------
// Struct containing acceptance rates of Metropolis and topological update

struct AcceptanceRates
{
    uint_fast64_t metro_acceptance;
    uint_fast64_t top_acceptance;
};

//----------------------------------------
// Enum to determine the order in which to go through the lattice (for example during updating)

enum class Order {ForwardsSeq, BackwardsSeq, ForwardsCheckerPar, BackwardsCheckerPar};

//----------------------------------------
// Enum to determine integration scheme used in Wilson flow

enum class WilsonFlowScheme {APE, Stout, StoutNew, EulerImplicit, EulerImplicitNew, MidpointImplicit, RK4};
// TODO: Like this, or maybe even simpler? Could reduce to {Cooling, Smearing, GradientFlow}
enum class SmoothinScheme {Cooling, APE, Stout, GFEulerExplicit, GFEulerImplicit, GFRK4};

#endif // LETTUCE_DRESSING_HPP
