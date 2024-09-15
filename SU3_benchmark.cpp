// Non-standard library headers
// Include these three header files first in this order
#include "LettuceGauge/defines.hpp"
#include "LettuceGauge/coords.hpp"
#include "LettuceGauge/lattice.hpp"
//-----
// Remaining files in alphabetic order (for now)
#include "LettuceGauge/actions/gauge/rectangular_action.hpp"
#include "LettuceGauge/IO/ansi_colors.hpp"
#include "LettuceGauge/IO/config_io/bmw_format.hpp"
#include "LettuceGauge/IO/config_io/bridge_text_format.hpp"
#include "LettuceGauge/IO/config_io/checkpoint_manager.hpp"
#include "LettuceGauge/IO/parameter_io.hpp"
#include "LettuceGauge/iterators/iterators.hpp"
#include "LettuceGauge/math/su2.hpp"
#include "LettuceGauge/math/su3.hpp"
#include "LettuceGauge/math/su3_exp.hpp"
#include "LettuceGauge/metadynamics.hpp"
// #include "LettuceGauge/observables/observables.hpp"
#include "LettuceGauge/observables/clover.hpp"
#include "LettuceGauge/observables/plaquette.hpp"
#include "LettuceGauge/observables/field_strength_tensor.hpp"
#include "LettuceGauge/observables/polyakov_loop.hpp"
#include "LettuceGauge/observables/topological_charge.hpp"
#include "LettuceGauge/observables/wilson_loop.hpp"
#include "LettuceGauge/smearing/cooling.hpp"
#include "LettuceGauge/smearing/stout_smearing.hpp"
#include "LettuceGauge/smearing/gradient_flow.hpp"
#include "LettuceGauge/updates/ghmc_gauge.hpp"
#include "LettuceGauge/updates/heatbath.hpp"
#include "LettuceGauge/updates/hmc_gauge.hpp"
#include "LettuceGauge/updates/hmc_metadynamics.hpp"
#include "LettuceGauge/updates/instanton.hpp"
#include "LettuceGauge/updates/metropolis.hpp"
#include "LettuceGauge/updates/overrelaxation.hpp"
#include "LettuceGauge/updates/tempering.hpp"
//-----
#include "PCG/pcg_random.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <algorithm>
#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>
//----------------------------------------
// Standard C headers
#include <cmath>
#include <ctime>

template<typename ClockT = std::chrono::steady_clock> //requires(std::chrono::is_clock_v<ClockT>)
struct Timer
{
    private:
        typename ClockT::time_point start_time;
    public:
        Timer() : start_time(ClockT::now()) {}

        void ResetTimer()
        {
            start_time = ClockT::now();
        }

        auto GetTime() const
        {
            return std::chrono::duration_cast<std::chrono::microseconds>(ClockT::now() - start_time).count();
        }
};

void FillVector(std::vector<Matrix_SU3>& vec)
{
    using size_type = typename std::vector<Matrix_SU3>::size_type;
    for (size_type i = 0; i < vec.size(); ++i)
    {
        // vec[i] = initial_value;
        // initial_value += Matrix_SU3(1);
        vec[i] = SU3::RandomMatrix(global_prng.UniformInt(), global_prng.UniformReal());
    }
}

template<typename ElementT>
void AddVectorsLoop(std::vector<ElementT>& res, const std::vector<ElementT>& lhs, const std::vector<ElementT>& rhs)
{
    using size_type = typename std::vector<ElementT>::size_type;
    #pragma omp parallel for schedule(static)
    for (size_type i = 0; i < res.size(); ++i)
    {
        res[i] = lhs[i] + rhs[i];
    }
}

template<typename ElementT>
void MultiplyVectorsLoop(std::vector<ElementT>& res, const std::vector<ElementT>& lhs, const std::vector<ElementT>& rhs)
{
    using size_type = typename std::vector<ElementT>::size_type;
    #pragma omp parallel for schedule(static)
    for (size_type i = 0; i < res.size(); ++i)
    {
        res[i] = lhs[i] * rhs[i];
    }
}

// Unused stuff from new code
// template<typename T>
// void FillLattice(LatticeBase<T>& lat, typename T::ElementType initial_value = typename T::ElementType(0))
// {
//     for (IndexType i = 0; i < lat.size(); ++i)
//     {
//         lat(i) = initial_value;
//         // SimdComplexDoubleVector tmp = Lettuce::Complex<double>(1.0, 1.0);
//         // initial_value += typename T::ElementType(tmp);
//         // // typename T::VectorType tmp = typename T::ScalarType(1.0, 1.0);
//         // // initial_value += tmp;
//         initial_value += typename T::ElementType(1);
//     }
// }

// template<typename T>
// void AddLattices(LatticeBase<T>& res, const LatticeBase<T>& lhs, const LatticeBase<T>& rhs)
// {
//     res.Derived() = lhs + rhs;
// }

// template<typename T>
// void MultiplyLattices(LatticeBase<T>& res, const LatticeBase<T>& lhs, const LatticeBase<T>& rhs)
// {
//     res.Derived() = lhs * rhs;
// }

void RunExpressionTemplateBenchmarks(std::size_t length_min, std::size_t length_increment, std::size_t length_max)
{
    int         benchmark_warmup      = 10;
    int         benchmark_repetitions = 100;
    // Operations required for complex 3x3 matrix multiplication (taking into account site-fusion/vectorization)
    int N = 3;
    double      matrix_flops          = (8 * N * N * N - 2 * N * N);
    std::cout << "Running SU(3) benchmarks\n";
    std::cout << "  Warmup iterations:    " << benchmark_warmup      << "\n"
              << "  Benchmark iterations: " << benchmark_repetitions << "\n" << std::endl;

    std::cout << "Testing std::vector (a = b * c)" << std::endl;
    for(std::size_t length = length_min; length <= length_max; length += length_increment)
    {
        std::size_t size = length * length * length * length;
        // Initialize
        std::vector<Matrix_SU3> res(size);
        std::vector<Matrix_SU3> lhs(size);
        std::vector<Matrix_SU3> rhs(size);
        FillVector(lhs);
        FillVector(rhs);
        // Warmup
        for (int count = 0; count < benchmark_warmup; ++count)
        {
            MultiplyVectorsLoop(res, lhs, rhs);
        }
        // Measure
        Timer timer;
        for (int count = 0; count < benchmark_repetitions; ++count)
        {
            MultiplyVectorsLoop(res, lhs, rhs);
        }
        double total_time_seconds = timer.GetTime() / 1e6;
        double single_time = total_time_seconds / benchmark_repetitions;
        std::cout << "Length: " << length << std::endl;
        std::cout << "GFLOP/s: " << size * matrix_flops / single_time / 1e9 << std::endl;
    }

    // std::cout << "Testing Lattice class (a = b * c)" << std::endl;
    // for(std::size_t length = length_min; length <= length_max; length += length_increment)
    // {
    //     std::size_t size = length * length * length * length / LatticeElementTraits<ElementT>::n_simd_lanes;
    //     // Initialize
    //     Lattice<ElementT, 1> res(size);
    //     Lattice<ElementT, 1> lhs(size);
    //     Lattice<ElementT, 1> rhs(size);
    //     FillLattice(lhs);
    //     FillLattice(rhs);
    //     // Warmup
    //     for (int count = 0; count < benchmark_warmup; ++count)
    //     {
    //         MultiplyLattices(res, lhs, rhs);
    //     }
    //     // Measure
    //     Timer timer;
    //     for (int count = 0; count < benchmark_repetitions; ++count)
    //     {
    //         MultiplyLattices(res, lhs, rhs);
    //     }
    //     double total_time_seconds = timer.GetTime() / 1e6;
    //     double single_time = total_time_seconds / benchmark_repetitions;
    //     std::cout << "Length: " << length << std::endl;
    //     std::cout << "GFLOP/s: " << size * matrix_flops / single_time / 1e9 << std::endl;
    // }

    // Lattice<ElementT, 1> res(size);
    // Lattice<ElementT, 1> lhs(size);
    // Lattice<ElementT, 1> rhs(size);
    // std::vector<ElementT> res_vec(size);
    // std::vector<ElementT> lhs_vec(size);
    // std::vector<ElementT> rhs_vec(size);
    // std::valarray<ElementT> res_varr(size);
    // std::valarray<ElementT> lhs_varr(size);
    // std::valarray<ElementT> rhs_varr(size);
    // FillLattice(lhs);
    // FillLattice(rhs);
    // for (int count = 0; count < benchmark_repetitions; ++count)
    // {
    //     // timer.ResetTimer();
    //     AddLattices(res, lhs, rhs);
    //     // std::cout << timer.GetTime() << std::endl;
    // }

    // // std::cout << "==================================================" << std::endl;
    // FillVector(lhs_vec);
    // FillVector(rhs_vec);
    // for (int count = 0; count < benchmark_repetitions; ++count)
    // {
    //     // timer.ResetTimer();
    //     AddVectorsLoop(res_vec, lhs_vec, rhs_vec);
    //     // std::cout << timer.GetTime() << std::endl;
    // }
    // // std::cout << "==================================================" << std::endl;
    // for (int count = 0; count < benchmark_repetitions; ++count)
    // {
    //     // timer.ResetTimer();
    //     AddVectorsOperators(res_vec, lhs_vec, rhs_vec);
    //     // std::cout << timer.GetTime() << std::endl;
    // }
    // // std::cout << "==================================================" << std::endl;
    // FillValarray(res_varr);
    // FillValarray(lhs_varr);
    // FillValarray(rhs_varr);
    // for (int count = 0; count < benchmark_repetitions; ++count)
    // {
    //     // timer.ResetTimer();
    //     AddValarrays(res_varr, lhs_varr, rhs_varr);
    //     // std::cout << timer.GetTime() << std::endl;
    // }
    // // std::cout << "==================================================" << std::endl;
    // Prevent compiler from optimizing out the operations
    // std::cout << res(3) << std::endl;
    // std::cout << res_varr[3] << std::endl;
    // std::cout << res_vec[3] << std::endl;
}

// void RunMatrixVectorBenchmarks(std::size_t length_min, std::size_t length_increment, std::size_t length_max)
// {
//     int         benchmark_warmup      = 10;
//     int         benchmark_repetitions = 100;

//     std::cout << "Running expression template benchmarks for " << type_name<ElementT>() << "\n";
//     std::cout << "  Warmup iterations:    " << benchmark_warmup      << "\n"
//               << "  Benchmark iterations: " << benchmark_repetitions << "\n" << std::endl;

//     using stype = typename ElementT::ScalarType;
//     using vtype = Vector<ElementT, NDim>;
//     using mtype = Matrix<ElementT, NDim>;

//     std::cout << "Testing Lattice class (scalar * vector)" << std::endl;
//     for(std::size_t length = length_min; length <= length_max; length += length_increment)
//     {
//         std::size_t size = length * length * length * length / LatticeElementTraits<ElementT>::n_simd_lanes;
//         // Initialize
//         Lattice<vtype, 1> res(size);
//         Lattice<vtype, 1> vec(size);
//         stype scalar = stype(rand());
//         FillLattice(vec);
//         // Warmup
//         for (int count = 0; count < benchmark_warmup; ++count)
//         {
//             res = scalar * vec;
//         }
//         // Measure
//         Timer timer;
//         for (int count = 0; count < benchmark_repetitions; ++count)
//         {
//             res = scalar * vec;
//         }
//         double total_time_seconds = timer.GetTime() / 1e6;
//         double single_time = total_time_seconds / benchmark_repetitions;
//         double scalar_vector_flops = (2 * NDim) * LatticeElementTraits<ElementT>::n_simd_lanes;
//         std::cout << "Length: " << length << std::endl;
//         std::cout << "GFLOP/s: " << size * scalar_vector_flops / single_time / 1e9 << std::endl;
//     }

//     std::cout << "Testing Lattice class (scalar * vector + vector)" << std::endl;
//     for(std::size_t length = length_min; length <= length_max; length += length_increment)
//     {
//         std::size_t size = length * length * length * length / LatticeElementTraits<ElementT>::n_simd_lanes;
//         // Initialize
//         Lattice<vtype, 1> res(size);
//         Lattice<vtype, 1> lhs(size);
//         Lattice<vtype, 1> rhs(size);
//         stype scalar = stype(rand());
//         FillLattice(lhs);
//         FillLattice(rhs);
//         // Warmup
//         for (int count = 0; count < benchmark_warmup; ++count)
//         {
//             res = scalar * lhs + rhs;
//         }
//         // Measure
//         Timer timer;
//         for (int count = 0; count < benchmark_repetitions; ++count)
//         {
//             res = scalar * lhs + rhs;
//         }
//         double total_time_seconds = timer.GetTime() / 1e6;
//         double single_time = total_time_seconds / benchmark_repetitions;
//         double scalar_vector_vector_flops = (4 * NDim) * LatticeElementTraits<ElementT>::n_simd_lanes;
//         std::cout << "Length: " << length << std::endl;
//         std::cout << "GFLOP/s: " << size * scalar_vector_vector_flops / single_time / 1e9 << std::endl;
//     }

//     std::cout << "Testing Lattice class (vector + vector)" << std::endl;
//     for(std::size_t length = length_min; length <= length_max; length += length_increment)
//     {
//         std::size_t size = length * length * length * length / LatticeElementTraits<ElementT>::n_simd_lanes;
//         // Initialize
//         Lattice<vtype, 1> res(size);
//         Lattice<vtype, 1> lhs(size);
//         Lattice<vtype, 1> rhs(size);
//         FillLattice(lhs);
//         FillLattice(rhs);
//         // Warmup
//         for (int count = 0; count < benchmark_warmup; ++count)
//         {
//             res = lhs + rhs;
//         }
//         // Measure
//         Timer timer;
//         for (int count = 0; count < benchmark_repetitions; ++count)
//         {
//             res = lhs + rhs;
//         }
//         double total_time_seconds = timer.GetTime() / 1e6;
//         double single_time = total_time_seconds / benchmark_repetitions;
//         double vector_vector_flops = (2 * NDim) * LatticeElementTraits<ElementT>::n_simd_lanes;
//         std::cout << "Length: " << length << std::endl;
//         std::cout << "GFLOP/s: " << size * vector_vector_flops / single_time / 1e9 << std::endl;
//     }

//     std::cout << "Testing Lattice class (matrix * vector)" << std::endl;
//     for(std::size_t length = length_min; length <= length_max; length += length_increment)
//     {
//         std::size_t size = length * length * length * length / LatticeElementTraits<ElementT>::n_simd_lanes;
//         // Initialize
//         Lattice<vtype, 1> res(size);
//         Lattice<mtype, 1> lhs(size);
//         Lattice<vtype, 1> rhs(size);
//         FillLattice(lhs);
//         FillLattice(rhs);
//         // Warmup
//         for (int count = 0; count < benchmark_warmup; ++count)
//         {
//             res = lhs * rhs;
//         }
//         // Measure
//         Timer timer;
//         for (int count = 0; count < benchmark_repetitions; ++count)
//         {
//             res = lhs * rhs;
//         }
//         double total_time_seconds = timer.GetTime() / 1e6;
//         double single_time = total_time_seconds / benchmark_repetitions;
//         double matrix_vector_flops = (8 * NDim * NDim - 2 * NDim * NDim) * LatticeElementTraits<ElementT>::n_simd_lanes;
//         std::cout << "Length: " << length << std::endl;
//         std::cout << "GFLOP/s: " << size * matrix_vector_flops / single_time / 1e9 << std::endl;
//     }

//     std::cout << "Testing Lattice class (matrix * matrix)" << std::endl;
//     for(std::size_t length = length_min; length <= length_max; length += length_increment)
//     {
//         std::size_t size = length * length * length * length / LatticeElementTraits<ElementT>::n_simd_lanes;
//         // Initialize
//         Lattice<mtype, 1> res(size);
//         Lattice<mtype, 1> lhs(size);
//         Lattice<mtype, 1> rhs(size);
//         FillLattice(lhs);
//         FillLattice(rhs);
//         // Warmup
//         for (int count = 0; count < benchmark_warmup; ++count)
//         {
//             res = lhs * rhs;
//         }
//         // Measure
//         Timer timer;
//         for (int count = 0; count < benchmark_repetitions; ++count)
//         {
//             res = lhs * rhs;
//         }
//         double total_time_seconds = timer.GetTime() / 1e6;
//         double single_time = total_time_seconds / benchmark_repetitions;
//         double matrix_matrix_flops = (8 * NDim * NDim * NDim - 2 * NDim * NDim) * LatticeElementTraits<ElementT>::n_simd_lanes;
//         std::cout << "Length: " << length << std::endl;
//         std::cout << "GFLOP/s: " << size * matrix_matrix_flops / single_time / 1e9 << std::endl;
//     }

//     // The values during this computation may overflow, so do not be surpised by NaNs when checking the values
//     std::cout << "Testing Lattice class (matrix * matrix in-place)" << std::endl;
//     for(std::size_t length = length_min; length <= length_max; length += length_increment)
//     {
//         std::size_t size = length * length * length * length / LatticeElementTraits<ElementT>::n_simd_lanes;
//         // Initialize
//         Lattice<mtype, 1> lhs(size);
//         Lattice<mtype, 1> rhs(size);
//         FillLattice(lhs);
//         FillLattice(rhs);
//         // Warmup
//         for (int count = 0; count < benchmark_warmup; ++count)
//         {
//             lhs = lhs * rhs;
//         }
//         // Measure
//         Timer timer;
//         for (int count = 0; count < benchmark_repetitions; ++count)
//         {
//             lhs = lhs * rhs;
//         }
//         double total_time_seconds = timer.GetTime() / 1e6;
//         double single_time = total_time_seconds / benchmark_repetitions;
//         double matrix_matrix_flops = (8 * NDim * NDim * NDim - 2 * NDim * NDim) * LatticeElementTraits<ElementT>::n_simd_lanes;
//         std::cout << "Length: " << length << std::endl;
//         std::cout << "GFLOP/s: " << size * matrix_matrix_flops / single_time / 1e9 << std::endl;
//     }
// }

int main([[maybe_unused]] int argc, [[maybe_unused]] char const *argv[])
{
    RunExpressionTemplateBenchmarks(8, 8, 40);
    // RunMatrixVectorBenchmarks(8, 8, 40);
    return 0;
}
