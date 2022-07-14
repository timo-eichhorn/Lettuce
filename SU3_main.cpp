// Pure SU(3) theory
// Debug Flags: -DDEBUG_MODE_TERMINAL, -DFIXED_SEED

// #define EIGEN_USE_MKL_ALL

// Non-standard library headers
#include "LettuceGauge/coords.hpp"
#include "LettuceGauge/defines.hpp"
#include "LettuceGauge/iterators/iterators.hpp"
#include "LettuceGauge/lattice.hpp"
#include "LettuceGauge/math/su2.hpp"
#include "LettuceGauge/math/su3.hpp"
#include "LettuceGauge/math/su3_exp.hpp"
#include "LettuceGauge/metadynamics.hpp"
#include "LettuceGauge/observables/plaquette.hpp"
#include "LettuceGauge/observables/polyakov_loop.hpp"
#include "LettuceGauge/observables/topological_charge.hpp"
#include "LettuceGauge/observables/wilson_loop.hpp"
#include "LettuceGauge/updates/heatbath.hpp"
#include "LettuceGauge/updates/hmc_gauge.hpp"
#include "LettuceGauge/updates/overrelaxation.hpp"
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
#include <array>
#include <chrono>
#include <complex>
// #include <experimental/iterator>
#include <filesystem>
#include <fstream>
// #include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <iomanip>
#include <iostream>
#include <iterator>
//----------------------------------------
// Standard C headers
#include <cmath>
#include <ctime>

//-----

// std::unique_ptr<Gl_Lattice> Gluon         {std::make_unique<Gl_Lattice>()};
// std::unique_ptr<Gl_Lattice> Gluonsmeared1 {std::make_unique<Gl_Lattice>()};
// std::unique_ptr<Gl_Lattice> Gluonsmeared2 {std::make_unique<Gl_Lattice>()};
// std::unique_ptr<Gl_Lattice> Gluonchain    {std::make_unique<Gl_Lattice>()};
// std::unique_ptr<Full_tensor> F_tensor     {std::make_unique<Full_tensor>()};
// std::unique_ptr<Full_tensor> Q_tensor     {std::make_unique<Full_tensor>()};

// GaugeField     Gluon         {Nt, Nx, Ny, Nz};
// GaugeField     Gluonsmeared1 {Nt, Nx, Ny, Nz};
// GaugeField     Gluonsmeared2 {Nt, Nx, Ny, Nz};
// GaugeField     Gluonchain    {Nt, Nx, Ny, Nz};

GaugeField                   Gluon;
GaugeField                   Gluonsmeared1;
GaugeField                   Gluonsmeared2;
// For now only necessarry for global Metadynamics
// TODO: Move to large if constexpr environment together with entire metadynamics code
GaugeField                   Gluonsmeared3;
GaugeField                   Gluonchain;
std::unique_ptr<Full_tensor> F_tensor      {std::make_unique<Full_tensor>()};
std::unique_ptr<Full_tensor> Q_tensor      {std::make_unique<Full_tensor>()};

//-----
// Overload << for vectors and arrays?

// template <typename T>
// std::ostream& operator<<(ostream& out, const std::vector<T>& container)
// {
//     out << "Container dump begins: ";
//     std::copy(container.cbegin(), container.cend(), std::ostream_iterator<T>(out, " "));
//     out << "\n";
//     return out;
// }

// template <typename T>
// std::ostream& operator<<(ostream& out, const std::array<T>& container)
// {
//     out << "Container dump begins: ";
//     std::copy(container.cbegin(), container.cend(), std::ostream_iterator<T>(out, " "));
//     out << "\n";
//     // std::copy(container.cbegin(), std::prev(container.cend()), std::ostream_iterator<T>(correlationlog, ","));
//     // correlationlog << Correlation_Function.back() << "\n";
//     return out;
// }

//-------------------------------------------------------------------------------------

//-----
// Function to get user input with error handling
// TODO: Constrain target to writeable range or something like that

template<typename T>
void ValidatedIn(const std::string& message, T& target)
{
    // Keep count of tries and abort after too many tries (e.g. important when using nohup)
    size_t count {0};
    while (std::cout << message << "\n" && !(std::cin >> target) && count < 10)
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input.\n";
        ++count;
    }
}

//-----
// Receives simulation parameters from user input

void Configuration()
{
    cout << "\n\nSU(3) theory simulation\n";
    cout << "Current version: " << program_version;
    cout << "\n\n----------------------------------------\n\n";
    // Get simulation parameters from user input
    ValidatedIn("Please enter beta: ", beta);
    ValidatedIn("Please enter n_run: ", n_run);
    ValidatedIn("Please enter expectation_period: ", expectation_period);
    n_run_inverse = 1.0 / static_cast<double>(n_run);
    if (n_metro != 0 && multi_hit != 0)
    {
        metro_norm = 1.0 / (Nt * Nx * Ny * Nz * 4.0 * n_metro * multi_hit);
    }
    cout << "\n" << "Gauge field precision: " << typeid(floatT).name() << "\n";
    cout << "beta is " << beta << ".\n";
    cout << "n_run is " << n_run << " and expectation_period is " << expectation_period << ".\n";
    cout << "n_metro is " << n_metro << ".\n";
    cout << "multi_hit is " << multi_hit << ".\n";
    cout << "metro_target_acceptance is " << metro_target_acceptance << ".\n";
    cout << "n_heatbath is " << n_heatbath << ".\n";
    cout << "n_hmc is " << n_hmc << ".\n";
    cout << "n_orelax is " << n_orelax << ".\n";
    cout << "metadynamics_enabled is " << metadynamics_enabled << ".\n";
    cout << "metapotential_updated is " << metapotential_updated << ".\n"; 
    //cout << "Overall, there are " <<  << " lattice sites.\n\n";
}

//-----
// Writes simulation parameters to files

void SaveParameters(std::string filename, const std::string& starttimestring)
{
    datalog.open(filename, std::fstream::out | std::fstream::app);
    datalog << program_version << "\n";
    datalog << "logfile\n\n";
    #ifdef DEBUG_MODE_TERMINAL
    datalog << "DEBUG_MODE_TERMINAL\n";
    #endif
    datalog << starttimestring << "\n";
    datalog << "START_PARAMS\n";
    datalog << "Gauge field precision = " << typeid(floatT).name() << "\n";
    datalog << "Nt = " << Nt << "\n";
    datalog << "Nx = " << Nx << "\n";
    datalog << "Ny = " << Ny << "\n";
    datalog << "Nz = " << Nz << "\n";
    datalog << "beta = " << beta << "\n";
    datalog << "n_run = " << n_run << "\n";
    datalog << "expectation_period = " << expectation_period << "\n";
    datalog << "n_smear = " << n_smear << "\n";
    datalog << "n_smear_skip = " << n_smear_skip << "\n";
    datalog << "rho_stout = " << rho_stout << "\n";
    datalog << "n_metro = " << n_metro << "\n";
    datalog << "multi_hit = " << multi_hit << "\n";
    datalog << "metro_target_acceptance = " << metro_target_acceptance << "\n";
    datalog << "n_heatbath = " << n_heatbath << "\n";
    datalog << "n_hmc = " << n_hmc << "\n";
    datalog << "n_orelax = " << n_orelax << "\n";
    datalog << "metadynamics_enabled = " << metadynamics_enabled << "\n";
    datalog << "metapotential_updated = " << metapotential_updated << "\n"; 
    datalog << "END_PARAMS\n" << endl;
    datalog.close();
    datalog.clear();
}

//-----
// Creates directories and files to store data

void CreateFiles()
{
    LatticeSizeString = to_string(Nx) + "x" + to_string(Ny) + "x" + to_string(Nz) + "x" + to_string(Nt);
    betaString = to_string(beta);
    directoryname_pre = "SU(3)_N=" + LatticeSizeString + "_beta=" + betaString;
    directoryname = directoryname_pre;
    append = 1;
    while (std::filesystem::exists(directoryname) == true)
    {
        appendString = to_string(append);
        directoryname = directoryname_pre + " (" + appendString + ")";
        ++append;
    }
    std::filesystem::create_directory(directoryname);
    cout << "\n\n" << "Created directory \"" << directoryname << "\".\n";
    logfilepath = directoryname + "/log.txt";
    parameterfilepath = directoryname + "/parameters.txt";
    wilsonfilepath = directoryname + "/wilson.txt";
    metapotentialfilepath = directoryname + "/metapotential.txt";
    cout << "Filepath (log):\t\t" << logfilepath << "\n";
    cout << "Filepath (parameters):\t" << parameterfilepath << "\n";
    cout << "Filepath (wilson):\t" << wilsonfilepath << "\n";
    cout << "Filepath (metadyn):\t" << metapotentialfilepath << "\n";
    #ifdef DEBUG_MODE_TERMINAL
    cout << "DEBUG_MODE_TERMINAL\n\n";
    #endif

    //-----
    // Writes parameters to files

    std::time_t start_time {std::chrono::system_clock::to_time_t(start)};
    std::string start_time_string {std::ctime(&start_time)};

    // logfile

    SaveParameters(logfilepath, start_time_string);

    // parameterfile

    SaveParameters(parameterfilepath, start_time_string);

    // wilsonfile

    SaveParameters(wilsonfilepath, start_time_string);
}

//-----
// Print final parameters to a specified ostream

template<typename floatT>
void PrintFinal(std::ostream& log, const uint_fast64_t acceptance_count, const uint_fast64_t acceptance_count_or, const uint_fast64_t acceptance_count_hmc, const floatT epsilon, const std::time_t& end_time, const std::chrono::duration<double>& elapsed_seconds)
{
    double or_norm {1.0};
    if constexpr(n_orelax != 0)
    {
        or_norm = 1.0 / (Nt * Nx * Ny * Nz * 4.0 * n_run * n_orelax);
    }
    double hmc_norm {1.0};
    if constexpr(n_hmc != 0)
    {
        hmc_norm = 1.0 / n_run;
    }
    log << "Metro target acceptance: " << metro_target_acceptance << "\n";
    log << "Metro acceptance: " << acceptance_count * metro_norm << "\n";
    log << "OR acceptance: " << acceptance_count_or * or_norm << "\n";
    log << "HMC acceptance: " << acceptance_count_hmc * hmc_norm << "\n";
    log << "epsilon: " << epsilon << "\n";
    log << std::ctime(&end_time) << "\n";
    log << "Required time: " << elapsed_seconds.count() << "s\n";
}

//-----
// Initialize multiple instances of PRNGs in vector for parallel usage

// void InitializePRNGs(vector<pcg64>& prng_vector)
// {
//     prng_vector.clear();
//     cout << "Beginning" << endl;
//     cout << "max threads:" << omp_get_max_threads() << endl;
//     array<array<int, 10>, 4> prng_test;
//     // array<int, 100> prng_test;
//     std::uniform_int_distribution<int> distribution_choice(1, 8);
//     for (int i = 0; i < omp_get_max_threads(); ++i)
//     {
//         // prng_vector.emplace_back(generator_rand);
//         pcg_extras::seed_seq_from<std::random_device> seed_source_temp;
//         #ifdef FIXED_SEED
//         pcg64 generator_rand_temp(i);
//         #else
//         pcg64 generator_rand_temp(seed_source_temp);
//         #endif
//         prng_vector.emplace_back(generator_rand_temp);
//     }

//     #pragma omp parallel for
//     for (int i = 0; i < 4; ++i)
//     {
//         pcg64& current_prng = prng_vector[omp_get_thread_num()];
//         for (int j = 0; j < 10; ++j)
//         {
//             // cout << "i: " << i << " j: " << j << endl;
//             prng_test[i][j] = distribution_choice(current_prng);
//         }
//     }
//     for (int i = 0; i < 4; ++i)
//     {
//         for (int j = 0; j < 10; ++j)
//         {
//             cout << "i:" << i << " j:" << j << " " << prng_test[i][j] << endl;
//         }
//     }
//     // cout << prng_vector.size() << endl;
//     // #pragma omp parallel for
//     // for (int i = 0; i < 100; ++i)
//     // {
//     //     pcg64& current_prng = prng_vector[omp_get_thread_num()];
//     //     prng_test[i] = distribution_choice(current_prng);
//     // }

//     // for (int i = 0; i < 100; ++i)
//     // {
//     //     cout << prng_test[i] << endl;
//     // }
// }

[[nodiscard]]
vector<pcg64> CreatePRNGs(const int thread_num = 0)
{
    vector<pcg64> temp_vec;
    #if defined(_OPENMP)
        int max_thread_num {omp_get_max_threads()};
    #else
        int max_thread_num {1};
    #endif
    cout << "Maximum number of threads: " << max_thread_num << endl;
    #if defined(_OPENMP)
        if (thread_num != 0)
        {
            max_thread_num = thread_num;
            omp_set_num_threads(thread_num);
        }
    #endif
    if (max_thread_num != 1)
    {
        cout << "Creating PRNG vector with " << max_thread_num << " PRNGs.\n" << endl;
    }
    else
    {
        cout << "Creating PRNG vector with " << max_thread_num << " PRNG.\n" << endl;
    }
    for (int thread_count = 0; thread_count < max_thread_num; ++thread_count)
    {
        #ifdef FIXED_SEED
        pcg64 generator_rand_temp(thread_count);
        temp_vec.emplace_back(generator_rand_temp);
        // temp_vec.emplace_back(generator_rand_temp(thread_count));
        #else
        pcg_extras::seed_seq_from<std::random_device> seed_source_temp;
        pcg64 generator_rand_temp(seed_source_temp);
        temp_vec.emplace_back(generator_rand_temp);
        // temp_vec.emplace_back(generator_rand_temp(seed_source_temp));
        #endif
    }
    return temp_vec;
}

//-----
// Create vector of normal_distribution generators with mean 0 and standard deviation 1 for HMC

[[nodiscard]]
vector<std::normal_distribution<floatT>> CreateNormal(const int thread_num = 0)
{
    vector<std::normal_distribution<floatT>> temp_vec;
    #if defined(_OPENMP)
        int max_thread_num {omp_get_max_threads()};
    #else
        int max_thread_num {1};
    #endif
    cout << "Maximum number of threads: " << max_thread_num << endl;
    #if defined(_OPENMP)
        if (thread_num != 0)
        {
            max_thread_num = thread_num;
            omp_set_num_threads(thread_num);
        }
    #endif
    if (max_thread_num != 1)
    {
        cout << "Creating vector of normal_distributions with " << max_thread_num << " normal_distributions.\n" << endl;
    }
    else
    {
        cout << "Creating vector of normal_distributions with " << max_thread_num << " normal_distributions.\n" << endl;
    }
    for (int thread_count = 0; thread_count < max_thread_num; ++thread_count)
    {
        std::normal_distribution<floatT> temp_dist{0, 1};
        temp_vec.emplace_back(temp_dist);
    }
    return temp_vec;
}

//-----
// Set gauge fields to identity

void SetGluonToOne(GaugeField& Gluon)
{
    #pragma omp parallel for
    // for (auto &ind0 : Gluon)
    // for (auto &ind1 : ind0)
    // for (auto &ind2 : ind1)
    // for (auto &ind3 : ind2)
    // for (auto &GluonMatrix : ind3)
    // {
    //     GluonMatrix.setIdentity();
    // }
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int mu = 0; mu < 4; ++mu)
    {
        Gluon({t, x, y, z, mu}).setIdentity();
    }
    cout << "Gauge Fields set to identity!" << endl;
}

//-----
// Generates random 3 x 3 matrices

[[nodiscard]]
Matrix_SU3 RandomSU3(std::uniform_int_distribution<int>& distribution_choice, std::uniform_real_distribution<floatT>& distribution_unitary)
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
            tmp << c_phi, i<floatT> * s_phi, 0.0,
                   i<floatT> * s_phi, c_phi, 0.0,
                   0.0, 0.0, 1.0;
        }
        break;
        case 2:
        {
            floatT s_phi {std::sin(phi)};
            floatT c_phi {std::cos(phi)};
            tmp << c_phi, -s_phi, 0.0,
                   s_phi, c_phi, 0.0,
                   0.0, 0.0, 1.0;
        }
        break;
        case 3:
        {
            std::complex<floatT> exp_I_phi {std::exp(i<floatT> * phi)};
            tmp << exp_I_phi, 0.0, 0.0,
                   0.0, conj(exp_I_phi), 0.0,
                   0.0, 0.0, 1.0;
        }
        break;
        case 4:
        {
            floatT s_phi {std::sin(phi)};
            floatT c_phi {std::cos(phi)};
            tmp << c_phi, 0.0, i<floatT> * s_phi,
                   0.0, 1.0, 0.0,
                   i<floatT> * s_phi, 0.0, c_phi;
        }
        break;
        case 5:
        {
            floatT s_phi {std::sin(phi)};
            floatT c_phi {std::cos(phi)};
            tmp << c_phi, 0.0, s_phi,
                   0.0, 1.0, 0.0,
                   -s_phi, 0.0, c_phi;
        }
        break;
        case 6:
        {
            floatT s_phi {std::sin(phi)};
            floatT c_phi {std::cos(phi)};
            tmp << 1.0, 0.0, 0.0,
                   0.0, c_phi, i<floatT> * s_phi,
                   0.0, i<floatT> * s_phi, c_phi;
        }
        break;
        case 7:
        {
            floatT s_phi {std::sin(phi)};
            floatT c_phi {std::cos(phi)};
            tmp << 1.0, 0.0, 0.0,
                   0.0, c_phi, -s_phi,
                   0.0, s_phi, c_phi;
        }
        break;
        case 8:
        {
            floatT phi_tilde {phi / static_cast<floatT>(std::sqrt(3))};
            std::complex<floatT> exp_phi_tilde {std::exp(i<floatT> * phi_tilde)};
            tmp << exp_phi_tilde, 0.0, 0.0,
                   0.0, exp_phi_tilde, 0.0,
                   // 0.0, 0.0, (std::complex<floatT>(1.0, 0))/(exp_phi_tilde * exp_phi_tilde);
                   // 0.0, 0.0, 1.0/(exp_phi_tilde * exp_phi_tilde);
                   0.0, 0.0, 1.0/(exp_phi_tilde * exp_phi_tilde);
        }
        break;
    }
    return tmp;
}

[[nodiscard]]
Matrix_SU3 RandomSU3Parallel(const int choice, const floatT phi)
{
    Matrix_SU3 tmp;

    switch(choice)
    {
        case 1:
        {
            floatT s_phi {std::sin(phi)};
            floatT c_phi {std::cos(phi)};
            tmp << c_phi, i<floatT> * s_phi, 0.0,
                   i<floatT> * s_phi, c_phi, 0.0,
                   0.0, 0.0, 1.0;
        }
        break;
        case 2:
        {
            floatT s_phi {std::sin(phi)};
            floatT c_phi {std::cos(phi)};
            tmp << c_phi, -s_phi, 0.0,
                   s_phi, c_phi, 0.0,
                   0.0, 0.0, 1.0;
        }
        break;
        case 3:
        {
            std::complex<floatT> exp_I_phi {std::exp(i<floatT> * phi)};
            tmp << exp_I_phi, 0.0, 0.0,
                   0.0, conj(exp_I_phi), 0.0,
                   0.0, 0.0, 1.0;
        }
        break;
        case 4:
        {
            floatT s_phi {std::sin(phi)};
            floatT c_phi {std::cos(phi)};
            tmp << c_phi, 0.0, i<floatT> * s_phi,
                   0.0, 1.0, 0.0,
                   i<floatT> * s_phi, 0.0, c_phi;
        }
        break;
        case 5:
        {
            floatT s_phi {std::sin(phi)};
            floatT c_phi {std::cos(phi)};
            tmp << c_phi, 0.0, s_phi,
                   0.0, 1.0, 0.0,
                   -s_phi, 0.0, c_phi;
        }
        break;
        case 6:
        {
            floatT s_phi {std::sin(phi)};
            floatT c_phi {std::cos(phi)};
            tmp << 1.0, 0.0, 0.0,
                   0.0, c_phi, i<floatT> * s_phi,
                   0.0, i<floatT> * s_phi, c_phi;
        }
        break;
        case 7:
        {
            floatT s_phi {std::sin(phi)};
            floatT c_phi {std::cos(phi)};
            tmp << 1.0, 0.0, 0.0,
                   0.0, c_phi, -s_phi,
                   0.0, s_phi, c_phi;
        }
        break;
        case 8:
        {
            floatT phi_tilde {phi / static_cast<floatT>(std::sqrt(3))};
            std::complex<floatT> exp_phi_tilde {std::exp(i<floatT> * phi_tilde)};
            tmp << exp_phi_tilde, 0.0, 0.0,
                   0.0, exp_phi_tilde, 0.0,
                   0.0, 0.0, 1.0/(exp_phi_tilde * exp_phi_tilde);
        }
        break;
    }
    return tmp;
}

//-----
// Returns unnormalized Lüscher-Weisz gauge action
// TODO: Implement

// [[nodiscard]]
// double LWGaugeAction(const GaugeField& Gluon)
// {

// }

//-----
// Cayley map to transform Lie algebra elements to the associated Lie group
// Can be used as an alernative to the exponential map in the HMC

[[nodiscard]]
Matrix_SU3 CayleyMap(Matrix_3x3 Mat)
{
    return (Matrix_SU3::Identity() - Mat).inverse() * (Matrix_SU3::Identity() + Mat);
}

//-----
// Stout smearing of gluon fields in all 4 directions

void StoutSmearing4D(const GaugeField& Gluon_unsmeared, GaugeField& Gluon_smeared, const floatT smear_param = 0.12)
{
    Matrix_3x3 Sigma;
    Matrix_3x3 A;
    Matrix_3x3 B;
    Matrix_3x3 C;

    // #pragma omp parallel for collapse(4) private(Sigma, A, B, C, D)
    #pragma omp parallel for private(Sigma, A, B, C)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        for (int mu = 0; mu < 4; ++mu)
        {
            // int coord {(((t * 32 + x) * 32 + y) * 32 + z) * 4 + mu};
            // int coord{t * 131072 + x * 4096 + y * 128 + z * 4 + mu};
            Sigma.noalias() = WilsonAction::Staple(Gluon_unsmeared, {t, x, y, z}, mu);
            // A.noalias() = Sigma * Gluon_unsmeared(t, x, y, z, mu).adjoint();
            A.noalias() = Sigma * Gluon_unsmeared({t, x, y, z, mu}).adjoint();
            B.noalias() = A - A.adjoint();
            C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
            // Gluon_smeared(t, x, y, z, mu) = (smear_param * C).exp() * Gluon_unsmeared(t, x, y, z, mu);
            Gluon_smeared({t, x, y, z, mu}) = SU3::exp(-i<floatT> * smear_param * C) * Gluon_unsmeared({t, x, y, z, mu});
            // Gluon_smeared[t][x][y][z][mu] = CayleyMap(i<floatT> * smear_param * C) * Gluon_unsmeared[t][x][y][z][mu];
            // ProjectionSU3Single(Gluon_smeared(t, x, y, z, mu));
            SU3::Projection::GramSchmidt(Gluon_smeared({t, x, y, z, mu}));
        }
    }
}

// [[nodiscard]]
// SmearedFieldTuple StoutSmearingN(GaugeField& Gluon1, GaugeField& Gluon2, const int n_smear, const floatT smear_param = 0.12)
// {
//     for (int smear_count = 0; smear_count < n_smear; ++smear_count)
//     {
//         if (smear_count % 2 == 0)
//         {
//             StoutSmearing4D(Gluon1, Gluon2, smear_param);
//         }
//         else
//         {
//             StoutSmearing4D(Gluon2, Gluon1, smear_param);
//         }
//     }
//     if (n_smear % 2 == 0)
//     {
//         return {Gluon1, Gluon2};
//     }
//     else
//     {
//         return {Gluon2, Gluon1};
//     }
// }

// TODO: This is potentially dangerous, since we need to make sure we use the correct Gluon array afterwards,
//       which depends on n_smear. For even n_smear, we need to use Gluon1, for odd n_smear we need to use Gluon2!

void StoutSmearingN(GaugeField& Gluon1, GaugeField& Gluon2, const int N, const floatT smear_param = 0.12)
{
    for (int smear_count = 0; smear_count < N; ++smear_count)
    {
        if (smear_count % 2 == 0)
        {
            StoutSmearing4D(Gluon1, Gluon2, smear_param);
        }
        else
        {
            StoutSmearing4D(Gluon2, Gluon1, smear_param);
        }
    }
}

// void StoutSmearingN(GaugeFieldSmeared& SmearedFields, const int offset, const int n_smear, const floatT smear_param = 0.12)
// {
//     for (int smear_count = 0; smear_count < n_smear; ++smear_count)
//     {
//         StoutSmearing4D(SmearedFields[(offset + smear_count) % 2], SmearedFields[(offset + smear_count + 1) % 2], smear_param);
//     }
// }

//-----
// Wilson flow using some integrator
// TODO: For now with fixed step-size, later implement adaptive step size?

// void WilsonFlow(const Gl_Lattice& Gluon, Gl_Lattice& Gluon_flowed, const double epsilon)
// {
//     Gluon_flowed = Gluon;
//     std::unique_pointer<Gl_Lattice> temp1 = std::make_unique<Gl_Lattice>();
//     for (int t_current = 0; t_current < t; t_current += epsilon)
//     {
//         for (int mu = 0; mu < 4; ++mu)
//         for (int t = 0; t < Nt; ++t)
//         for (int x = 0; x < Nx; ++x)
//         for (int y = 0; y < Ny; ++y)
//         for (int z = 0; z < Nz; ++z)
//         {
//             temp1[t][x][y][z] = epsilon * ;
//         }
//         // TODO: Implement me
//         // TODO: Save the sum of Z_1 and Z_0 to use twice
//         Gluon_flowed = (3.0/4.0 * Z_2 - 8.0/9.0 * Z_1 + 17.0/36.0 * Z_0).exp() * (8.0/9.0 * Z_1 - 17.0/36.0 * Z_0).exp() * (1.0/4.0 * Z_0).exp() * Gluon_flowed;
//     }
// }

void WilsonFlowForward(GaugeField& Gluon, const double epsilon, const int n_flow)
{
    Matrix_3x3 st;
    Matrix_3x3 A;
    Matrix_3x3 B;
    Matrix_3x3 C;

    #if defined(_OPENMP)
    // Parallel version
    for (int flow_count = 0; flow_count < n_flow; ++flow_count)
    {
        for (int eo = 0; eo < 2; ++eo)
        for (int mu = 0; mu < 4; ++mu)
        {
            #pragma omp parallel for private(st, A, B, C)
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            {
                int offset {((t + x + y) & 1) ^ eo};
                for (int z = offset; z < Nz; z+=2)
                {
                    st.noalias() = WilsonAction::Staple(Gluon, {t, x, y, z}, mu);
                    A.noalias() = st * Gluon({t, x, y, z, mu}).adjoint();
                    B.noalias() = A - A.adjoint();
                    C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
                    // Gluon({t, x, y, z, mu}) = (epsilon * C).exp() * Gluon({t, x, y, z, mu});
                    Gluon({t, x, y, z, mu}) = SU3::exp(-i<floatT> * epsilon * C) * Gluon({t, x, y, z, mu});
                    //-----
                    SU3::Projection::GramSchmidt(Gluon({t, x, y, z, mu}));
                }
            }
        }
    }
    #else
    // Sequential version
    for (int flow_count = 0; flow_count < n_flow; ++flow_count)
    {
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        for (int z = 0; z < Nz; ++z)
        {
            for (int mu = 0; mu < 4; ++mu)
            {
                st.noalias() = WilsonAction::Staple(Gluon, {t, x, y, z}, mu);
                A.noalias() = st * Gluon({t, x, y, z, mu}).adjoint();
                B.noalias() = A - A.adjoint();
                C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
                // Gluon({t, x, y, z, mu}) = (epsilon * C).exp() * Gluon({t, x, y, z, mu});
                Gluon({t, x, y, z, mu}) = SU3::exp(-i<floatT> * epsilon * C) * Gluon({t, x, y, z, mu});
                //-----
                SU3::Projection::GramSchmidt(Gluon({t, x, y, z, mu}));
            }
        }
    }
    #endif
}

// Seems to be somewhat invertible if precision = 1e-12, but a precision of 1e-8 is definitely not sufficient

void WilsonFlowBackward(GaugeField& Gluon, GaugeField& Gluon_temp, const double epsilon, const int n_flow, const floatT precision = 1e-14)
{
    Gluon_temp = Gluon;
    Matrix_3x3 st;
    Matrix_3x3 A;
    Matrix_3x3 B;
    Matrix_3x3 C;
    Matrix_SU3 old_link;

    #if defined (_OPENMP)
    // Parallel version
    for (int flow_count = 0; flow_count < n_flow; ++flow_count)
    {
        for (int eo = 1; eo >= 0; --eo)
        for (int mu = 3; mu >= 0; --mu)
        {
            #pragma omp parallel for private(st, A, B, C, old_link)
            for (int t = Nt - 1; t >= 0; --t)
            for (int x = Nx - 1; x >= 0; --x)
            for (int y = Ny - 1; y >= 0; --y)
            {
                int offset {((t + x + y) & 1) ^ eo};
                for (int z = Nz - 1 - (1 + Nz%2 + offset)%2; z >= offset; z-=2)
                {
                    do
                    {
                        old_link = Gluon_temp({t, x, y, z, mu});
                        SU3::Projection::GramSchmidt(old_link);
                        st.noalias() = WilsonAction::Staple(Gluon_temp, {t, x, y, z}, mu);
                        A.noalias() = st * Gluon_temp({t, x, y, z, mu}).adjoint();
                        B.noalias() = A - A.adjoint();
                        C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
                        // Gluon_temp({t, x, y, z, mu}) = (epsilon * C).exp() * Gluon({t, x, y, z, mu});
                        Gluon_temp({t, x, y, z, mu}) = SU3::exp(-i<floatT> * epsilon * C) * Gluon({t, x, y, z, mu});
                        //-----
                        SU3::Projection::GramSchmidt(Gluon_temp({t, x, y, z, mu}));
                    }
                    while ((Gluon_temp({t, x, y, z, mu}) - old_link).norm() > precision);
                }
            }
        }
        // Copy to original array
        Gluon = Gluon_temp;
    }
    #else
    // Sequential version
    for (int flow_count = 0; flow_count < n_flow; ++flow_count)
    {
        for (int t = Nt - 1; t >= 0; --t)
        for (int x = Nx - 1; x >= 0; --x)
        for (int y = Ny - 1; y >= 0; --y)
        for (int z = Nz - 1; z >= 0; --z)
        {
            for (int mu = 3; mu >= 0; --mu)
            {
                do
                {
                    old_link = Gluon_temp({t, x, y, z, mu});
                    SU3::Projection::GramSchmidt(old_link);
                    st.noalias() = WilsonAction::Staple(Gluon_temp, {t, x, y, z}, mu);
                    A.noalias() = st * Gluon_temp({t, x, y, z, mu}).adjoint();
                    B.noalias() = A - A.adjoint();
                    C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
                    // Gluon_temp({t, x, y, z, mu}) = (epsilon * C).exp() * Gluon({t, x, y, z, mu});
                    Gluon_temp({t, x, y, z, mu}) = SU3::exp(-i<floatT> * epsilon * C) * Gluon({t, x, y, z, mu});
                    //-----
                    SU3::Projection::GramSchmidt(Gluon_temp({t, x, y, z, mu}));
                }
                while ((Gluon_temp({t, x, y, z, mu}) - old_link).norm() > precision);
            }
        }
        // Copy to original array
        Gluon = Gluon_temp;
    }
    #endif
}

//-----
// Calculates clover term

// template<int Nmu>
// void Clover(const Gl_Lattice& Gluon, Full_tensor& Q)
void Clover(const GaugeField& Gluon, GaugeField& Gluonchain, Full_tensor& Clov, const int Nmu)
{
    // Gl_Lattice Gluonchain;
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        for (int mu = 0; mu < 4; ++mu)
        {
            Gluonchain({t, x, y, z, mu}).setIdentity();
        }
        for (int n = 0; n < Nmu; ++ n)
        {
            Gluonchain({t, x, y, z, 0}) *= Gluon({(t + n)%Nt, x, y, z, 0});
            Gluonchain({t, x, y, z, 1}) *= Gluon({t, (x + n)%Nx, y, z, 1});
            Gluonchain({t, x, y, z, 2}) *= Gluon({t, x, (y + n)%Ny, z, 2});
            Gluonchain({t, x, y, z, 3}) *= Gluon({t, x, y, (z + n)%Nz, 3});
        }
    }

    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        int tm = (t - Nmu + Nt)%Nt;
        int xm = (x - Nmu + Nx)%Nx;
        int ym = (y - Nmu + Ny)%Ny;
        int zm = (z - Nmu + Nz)%Nz;
        int tp = (t + Nmu)%Nt;
        int xp = (x + Nmu)%Nx;
        int yp = (y + Nmu)%Ny;
        int zp = (z + Nmu)%Nz;

        Clov[t][x][y][z][0][0].setZero();
        Clov[t][x][y][z][0][1] = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 1}) * Gluon({t, xp, y, z, 0}).adjoint() * Gluon({t, x, y, z, 1}).adjoint()
                               + Gluon({t, x, y, z, 1}) * Gluon({tm, xp, y, z, 0}).adjoint() * Gluon({tm, x, y, z, 1}).adjoint() * Gluon({tm, x, y, z, 0})
                               + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, xm, y, z, 1}).adjoint() * Gluon({tm, xm, y, z, 0}) * Gluon({t, xm, y, z, 1})
                               + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, z, 0}) * Gluon({tp, xm, y, z, 1}) * Gluon({t, x, y, z, 0}).adjoint();
        Clov[t][x][y][z][1][0] = Clov[t][x][y][z][0][1].adjoint();

        Clov[t][x][y][z][0][2] = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 2}) * Gluon({t, x, yp, z, 0}).adjoint() * Gluon({t, x, y, z, 2}).adjoint()
                               + Gluon({t, x, y, z, 2}) * Gluon({tm, x, yp, z, 0}).adjoint() * Gluon({tm, x, y, z, 2}).adjoint() * Gluon({tm, x, y, z, 0})
                               + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, ym, z, 2}).adjoint() * Gluon({tm, x, ym, z, 0}) * Gluon({t, x, ym, z, 2})
                               + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 0}) * Gluon({tp, x, ym, z, 2}) * Gluon({t, x, y, z, 0}).adjoint();
        Clov[t][x][y][z][2][0] = Clov[t][x][y][z][0][2].adjoint();

        Clov[t][x][y][z][0][3] = Gluon({t, x, y, z, 0}) * Gluon({tp, x, y, z, 3}) * Gluon({t, x, y, zp, 0}).adjoint() * Gluon({t, x, y, z, 3}).adjoint()
                               + Gluon({t, x, y, z, 3}) * Gluon({tm, x, y, zp, 0}).adjoint() * Gluon({tm, x, y, z, 3}).adjoint() * Gluon({tm, x, y, z, 0})
                               + Gluon({tm, x, y, z, 0}).adjoint() * Gluon({tm, x, y, zm, 3}).adjoint() * Gluon({tm, x, y, zm, 0}) * Gluon({t, x, y, zm, 3})
                               + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 0}) * Gluon({tp, x, y, zm, 3}) * Gluon({t, x, y, z, 0}).adjoint();
        Clov[t][x][y][z][3][0] = Clov[t][x][y][z][0][3].adjoint();

        Clov[t][x][y][z][1][1].setZero();
        Clov[t][x][y][z][1][2] = Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 2}) * Gluon({t, x, yp, z, 1}).adjoint() * Gluon({t, x, y, z, 2}).adjoint()
                               + Gluon({t, x, y, z, 2}) * Gluon({t, xm, yp, z, 1}).adjoint() * Gluon({t, xm, y, z, 2}).adjoint() * Gluon({t, xm, y, z, 1})
                               + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, ym, z, 2}).adjoint() * Gluon({t, xm, ym, z, 1}) * Gluon({t, x, ym, z, 2})
                               + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, z, 1}) * Gluon({t, xp, ym, z, 2}) * Gluon({t, x, y, z, 1}).adjoint();
        Clov[t][x][y][z][2][1] = Clov[t][x][y][z][1][2].adjoint();

        Clov[t][x][y][z][1][3] = Gluon({t, x, y, z, 1}) * Gluon({t, xp, y, z, 3}) * Gluon({t, x, y, zp, 1}).adjoint() * Gluon({t, x, y, z, 3}).adjoint()
                               + Gluon({t, x, y, z, 3}) * Gluon({t, xm, y, zp, 1}).adjoint() * Gluon({t, xm, y, z, 3}).adjoint() * Gluon({t, xm, y, z, 1})
                               + Gluon({t, xm, y, z, 1}).adjoint() * Gluon({t, xm, y, zm, 3}).adjoint() * Gluon({t, xm, y, zm, 1}) * Gluon({t, x, y, zm, 3})
                               + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 1}) * Gluon({t, xp, y, zm, 3}) * Gluon({t, x, y, z, 1}).adjoint();
        Clov[t][x][y][z][3][1] = Clov[t][x][y][z][1][3].adjoint();

        Clov[t][x][y][z][2][2].setZero();
        Clov[t][x][y][z][2][3] = Gluon({t, x, y, z, 2}) * Gluon({t, x, yp, z, 3}) * Gluon({t, x, y, zp, 2}).adjoint() * Gluon({t, x, y, z, 3}).adjoint()
                               + Gluon({t, x, y, z, 3}) * Gluon({t, x, ym, zp, 2}).adjoint() * Gluon({t, x, ym, z, 3}).adjoint() * Gluon({t, x, ym, z, 2})
                               + Gluon({t, x, ym, z, 2}).adjoint() * Gluon({t, x, ym, zm, 3}).adjoint() * Gluon({t, x, ym, zm, 2}) * Gluon({t, x, y, zm, 3})
                               + Gluon({t, x, y, zm, 3}).adjoint() * Gluon({t, x, y, zm, 2}) * Gluon({t, x, yp, zm, 3}) * Gluon({t, x, y, z, 2}).adjoint();
        Clov[t][x][y][z][3][2] = Clov[t][x][y][z][2][3].adjoint();
        Clov[t][x][y][z][3][3].setZero();
    }
}

//-----
// Calculates the field strength tensor using clover definition

void Fieldstrengthtensor(const GaugeField& Gluon, GaugeField& Gluonchain, Full_tensor& F, Full_tensor& Q, const int Nmu)
{
    Clover(Gluon, Gluonchain, Q, Nmu);

    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int mu = 0; mu < 4; ++mu)
    for (int nu = 0; nu < 4; ++nu)
    {
        // F[t][x][y][z][mu][nu] = std::complex<floatT> (0.0, - 0.125) * (Q[t][x][y][z][mu][nu] - Q[t][x][y][z][nu][mu]);
        // F[t][x][y][z][mu][nu] = std::complex<floatT> (0.0, - 1.0 / (8 * Nmu * Nmu)) * (Q[t][x][y][z][mu][nu] - Q[t][x][y][z][nu][mu]);
        // F[t][x][y][z][mu][nu] = -i<floatT>/(8.0 * Nmu * Nmu) * (Q[t][x][y][z][mu][nu] - Q[t][x][y][z][nu][mu]);
        F[t][x][y][z][mu][nu] = -i<floatT>/(8.f * Nmu * Nmu) * (Q[t][x][y][z][mu][nu] - Q[t][x][y][z][nu][mu]);
    }

    // TODO: Rewrite like this
    // Local_tensor Q;
    // #pragma omp parallel for
    // for (int t = 0; t < Nt; ++t)
    // for (int x = 0; x < Nx; ++x)
    // for (int y = 0; y < Ny; ++y)
    // for (int z = 0; z < Nz; ++z)
    // {
    //     // Insert clover calculation here
    //     for (int mu = 0; mu < 4; ++mu)
    //     for (int nu = 0; nu < 4; ++nu)
    //     {
    //         // F[t][x][y][z][mu][nu] = std::complex<floatT> (0.0, - 0.125) * (Q[t][x][y][z][mu][nu] - Q[t][x][y][z][nu][mu]);
    //         // F[t][x][y][z][mu][nu] = std::complex<floatT> (0.0, - 1.0 / (8 * Nmu * Nmu)) * (Q[t][x][y][z][mu][nu] - Q[t][x][y][z][nu][mu]);
    //         // F[t][x][y][z][mu][nu] = -i<floatT>/(8.0 * Nmu * Nmu) * (Q[t][x][y][z][mu][nu] - Q[t][x][y][z][nu][mu]);
    //         F[t][x][y][z][mu][nu] = -i<floatT>/8.f * (Q[mu][nu] - Q[nu][mu]);
    //     }
    // }
}

//-----
// Calculates energy density from field strength tensor

[[nodiscard]]
double Energy_density(const GaugeField& Gluon, GaugeField& Gluonchain, Full_tensor& F, Full_tensor& Q, const int Nmu)
{
    cout << "\nBeginning of function" << endl;
    double e_density {0.0};

    cout << "\nCalculating field strength tensor" << endl;
    Fieldstrengthtensor(Gluon, Gluonchain, F, Q, Nmu);
    cout << "\nCalculated field strength tensor" << endl;

    #pragma omp parallel for reduction(+:e_density)
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int mu = 0; mu < 4; ++mu)
    for (int nu = 0; nu < 4; ++nu)
    {
        e_density += std::real((F[t][x][y][z][mu][nu] * F[t][x][y][z][mu][nu]).trace());
    }
    // e_density *= 1.0/(2.0 * Nx * Ny * Nz * Nt * Nmu * Nmu);
    e_density *= 1.0/(2.0 * 9.0 * Nx * Ny * Nz * Nt);
    // cout << e_density << "\n";
    return e_density;
}

//-----
// Metropolis update routine

void MetropolisUpdate(GaugeField& Gluon, const int n_sweep, uint_fast64_t& acceptance_count, floatT& epsilon, std::uniform_real_distribution<floatT>& distribution_prob, std::uniform_int_distribution<int>& distribution_choice, std::uniform_real_distribution<floatT>& distribution_unitary)
{
    acceptance_count = 0;

    // std::chrono::duration<double> staple_time {0.0};
    // std::chrono::duration<double> local_time {0.0};
    // std::chrono::duration<double> multihit_time {0.0};
    // std::chrono::duration<double> accept_reject_time {0.0};
    for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
    for (int eo = 0; eo < 2; ++eo)
    for (int mu = 0; mu < 4; ++mu)
    {
        // #pragma omp parallel for reduction(+:acceptance_count) shared(prng_vector) private(st, old_link, new_link, s, sprime) firstprivate(eo, mu)
        #pragma omp parallel for reduction(+:acceptance_count) shared(prng_vector)
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        {
            int offset {((t + x + y) & 1) ^ eo};
            for (int z = offset; z < Nz; z+=2)
            {
                // auto start_staple = std::chrono::high_resolution_clock::now();
                Matrix_3x3 st {WilsonAction::Staple(Gluon, {t, x, y, z}, mu)};
                // auto end_staple = std::chrono::high_resolution_clock::now();
                // staple_time += end_staple - start_staple;

                // auto start_local = std::chrono::high_resolution_clock::now();
                Matrix_SU3 old_link {Gluon({t, x, y, z, mu})};
                double s {WilsonAction::Local(old_link, st)};
                // auto end_local = std::chrono::high_resolution_clock::now();
                // local_time += end_local - start_local;
                // std::array<int, multi_hit>    prng_choice_vec;
                // std::array<floatT, multi_hit> prng_unitary_vec;
                // std::array<floatT, multi_hit> prng_prob_vec;
                // for (int n_hit = 0; n_hit < multi_hit; ++n_hit)
                // {
                //     prng_choice_vec[n_hit] = distribution_choice(prng_vector[omp_get_thread_num()]);
                //     prng_unitary_vec[n_hit] = distribution_unitary(prng_vector[omp_get_thread_num()]);
                //     prng_prob_vec[n_hit] = distribution_prob(prng_vector[omp_get_thread_num()]);
                // }
                for (int n_hit = 0; n_hit < multi_hit; ++n_hit)
                {
                    #if defined(_OPENMP)
                    // int choice = prng_choice_vec[n_hit];
                    // floatT phi = prng_unitary_vec[n_hit];
                    int choice {distribution_choice(prng_vector[omp_get_thread_num()])};
                    floatT phi {distribution_unitary(prng_vector[omp_get_thread_num()])};
                    Matrix_SU3 new_link {old_link * RandomSU3Parallel(choice, phi)};
                    #else
                    // auto start_multihit = std::chrono::high_resolution_clock::now();
                    int choice {distribution_choice(generator_rand)};
                    floatT phi {distribution_unitary(generator_rand)};
                    Matrix_SU3 new_link {old_link * RandomSU3Parallel(choice, phi)};
                    // auto end_multihit = std::chrono::high_resolution_clock::now();
                    // multihit_time += end_multihit - start_multihit;
                    #endif

                    // auto start_accept_reject = std::chrono::high_resolution_clock::now();
                    double sprime {WilsonAction::Local(new_link, st)};
                    double p {std::exp(-sprime + s)};
                    // TODO: Does this help in any way? Also try out for Orelax
                    // double p {std::exp(SLocalDiff(old_link - new_link, st))};
                    #if defined(_OPENMP)
                    double q {distribution_prob(prng_vector[omp_get_thread_num()])};
                    // double q = prng_prob_vec[n_hit];
                    #else
                    double q {distribution_prob(generator_rand)};
                    #endif

                    // Ugly hack to avoid branches in parallel region
                    // CAUTION: We would want to check if q <= p, since for beta = 0 everything should be accepted
                    // Unfortunately signbit(0) returns false... Is there way to fix this?
                    // bool accept {std::signbit(q - p)};
                    // Gluon[t][x][y][z][mu] = accept * new_link + (!accept) * old_link;
                    // old_link = accept * new_link + (!accept) * old_link;
                    // s = accept * sprime + (!accept) * s;
                    // acceptance_count += accept;
                    if (q <= p)
                    {
                        Gluon({t, x, y, z, mu}) = new_link;
                        old_link = new_link;
                        s = sprime;
                        acceptance_count += 1;
                    }
                    // auto end_accept_reject = std::chrono::high_resolution_clock::now();
                    // accept_reject_time += end_accept_reject - start_accept_reject;
                }
                SU3::Projection::GramSchmidt(Gluon({t, x, y, z, mu}));
            }
        }
    }
    // TODO: Test which acceptance rate is best. Initially had 0.8 as target, but 0.5 seems to thermalize much faster!
    // Adjust PRNG width to target mean acceptance rate of 0.5
    epsilon += (acceptance_count * metro_norm - static_cast<floatT>(metro_target_acceptance)) * static_cast<floatT>(0.2);
    // cout << "staple_time: " << staple_time.count() << "\n";
    // cout << "local_time: " << local_time.count() << "\n";
    // cout << "multihit_time: " << multihit_time.count() << "\n";
    // cout << "accept_reject_time: " << accept_reject_time.count() << endl;
}

//-----
// Overrelaxation update for SU(2)

template<typename floatT>
SU2_comp<floatT> OverrelaxationSU2Old(const SU2_comp<floatT>& A)
{
    floatT a_norm {static_cast<floatT>(1.0) / std::sqrt(A.det_sq())};
    SU2_comp V {a_norm * A};
    return (V * V).adjoint();
}

//-----
// Overrelaxation update for SU(3) using Cabibbo-Marinari method

void OverrelaxationSubgroupOld(GaugeField& Gluon, const int n_sweep)
{
    for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
    for (int mu = 0; mu < 4; ++mu)
    for (int eo = 0; eo < 2; ++eo)
    {
        #pragma omp parallel for
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        {
            int offset {((t + x + y) & 1) ^ eo};
            for (int z = offset; z < Nz; z+=2)
            {
                Matrix_3x3 W;
                SU2_comp<floatT> subblock;
                // Note: Our staple definition corresponds to the daggered staple in Gattringer & Lang, therefore use adjoint
                Matrix_3x3 st_adj {(WilsonAction::Staple(Gluon, {t, x, y, z}, mu)).adjoint()};
                //-----
                // Update (0, 1) subgroup
                // W = Gluon[t][x][y][z][mu] * st_adj;
                // std::cout << "Action before: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
                subblock = Extract01<floatT>(Gluon({t, x, y, z, mu}) * st_adj);
                Gluon({t, x, y, z, mu}) = Embed01(OverrelaxationSU2Old(subblock)) * Gluon({t, x, y, z, mu});
                // std::cout << "Action after: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
                //-----
                // Update (0, 2) subgroup
                // W = Gluon[t][x][y][z][mu] * st_adj;
                // std::cout << "Action before: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
                subblock = Extract02<floatT>(Gluon({t, x, y, z, mu}) * st_adj);
                Gluon({t, x, y, z, mu}) = Embed02(OverrelaxationSU2Old(subblock)) * Gluon({t, x, y, z, mu});
                // std::cout << "Action after: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
                //-----
                // Update (1, 2) subgroup
                // W = Gluon[t][x][y][z][mu] * st_adj;
                // std::cout << "Action before: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
                subblock = Extract12<floatT>(Gluon({t, x, y, z, mu}) * st_adj);
                Gluon({t, x, y, z, mu}) = Embed12(OverrelaxationSU2Old(subblock)) * Gluon({t, x, y, z, mu});
                // std::cout << "Action after: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
                //-----
                // Project link to SU(3)
                SU3::Projection::GramSchmidt(Gluon({t, x, y, z, mu}));
            }
        }
    }
}

//-----
// Heatbath update for SU(2)

template<typename floatT>
SU2_comp<floatT> HeatbathSU2(const SU2_comp<floatT>& A, const floatT prefactor, std::uniform_real_distribution<floatT>& distribution_uniform, const int max_iteration = 10)
{
    // Determinant of staple as norm to project staple back to SU(2)
    floatT a_norm {static_cast<floatT>(1.0) / std::sqrt(A.det_sq())};
    SU2_comp<floatT> V {a_norm * A};
    SU2_comp<floatT> mat_su2;
    floatT r1, r2, r3, x1, x2, x3, lambda_sq, r0;
    int count {0};
    do
    {
        // Generate random number lambda_sq following a polynomially modified Gaussian distribution (cf. Gattringer & Lang (4.43))
        // floatT r1 {static_cast<floatT>(1.0) - distribution_uniform(prng_vector[omp_get_thread_num()])};
        // floatT x1 {std::log(r1)};
        // floatT r2 {static_cast<floatT>(1.0) - distribution_uniform(prng_vector[omp_get_thread_num()])};
        // floatT x2 {std::cos(static_cast<floatT>(2.0) * pi<floatT> * r2)};
        // floatT r3 {static_cast<floatT>(1.0) - distribution_uniform(prng_vector[omp_get_thread_num()])};
        // floatT x3 {std::log(r3)};
        r1 = static_cast<floatT>(1.0) - distribution_uniform(prng_vector[omp_get_thread_num()]);
        x1 = std::log(r1);
        r2 = static_cast<floatT>(1.0) - distribution_uniform(prng_vector[omp_get_thread_num()]);
        x2 = std::cos(static_cast<floatT>(2.0) * pi<floatT> * r2);
        r3 = static_cast<floatT>(1.0) - distribution_uniform(prng_vector[omp_get_thread_num()]);
        x3 = std::log(r3);
        // Factor 0.25, so for N_col = 2 we get a factor 0.5, while for N_col = 3 we get a factor 0.75
        // floatT lambda_sq {static_cast<floatT>(-0.25 * prefactor * a_norm) * (x1 + x2 * x2 * x3)};
        lambda_sq = static_cast<floatT>(-0.25 * prefactor * a_norm) * (x1 + x2 * x2 * x3);
        //-----
        // Correct for factor sqrt(1 - lambda_sq) in probability distribution via accept-reject step
        // floatT r0 {distribution_uniform(prng_vector[omp_get_thread_num()])};
        r0 = distribution_uniform(prng_vector[omp_get_thread_num()]);
        if (count > max_iteration)
        {
            return {1.0, 0.0};
        }
        ++count;
    }
    while (r0 * r0 + lambda_sq >= static_cast<floatT>(1.0));
    // if (count > 3)
    // {
    //     std::cout << count << std::endl;
    // }

    // if (r0 * r0 + lambda_sq < static_cast<floatT>(1.0))
    // {
        // Calculate zeroth coefficient of our SU(2) matrix in quaternionic representation
        floatT x0 {static_cast<floatT>(1.0) - static_cast<floatT>(2.0) * lambda_sq};
        // Calculate absolute value of our random vector
        floatT abs_x {std::sqrt(static_cast<floatT>(1.0) - x0 * x0)};
        // Generate angular variables, i.e., random vector with length abs_x (simply generating three uniformly distributed values in the unit range and
        // normalizing them does not work, since the resulting distribution is not uniform, but biased against vectors close to the coordinate axes)
        // Instead, we generate a random vector with length abs_x in spherical coordinates
        // Since the functional determinant contains a factor sin(theta), directly generating the coordinates does not give a uniform distribution
        // Therefore, we generate cos(theta) in [-1, 1] using sqrt(1 - rand^2)
        // TODO: We want a random number in the closed interval [-1, 1], but the standard distribution only covers the half open interval [-1, 1) or
        // (-1, 1]. Therefore, do something like this: std::uniform_real_distribution<floatT> distribution_uniform(-1.0, std::nextafter(1.0, 2.0))
        // floatT r1 {static_cast<floatT>(1.0) - static_cast<floatT>(2.0) * distribution_uniform(prng_vector[omp_get_thread_num()])};
        // Random number in interval [0, 1)
        floatT phi {distribution_uniform(prng_vector[omp_get_thread_num()])};
        // Random number in interval (-1, 1]
        floatT cos_theta {static_cast<floatT>(1.0) - static_cast<floatT>(2.0) * distribution_uniform(prng_vector[omp_get_thread_num()])};
        floatT vec_norm {abs_x * std::sqrt(static_cast<floatT>(1.0) - cos_theta * cos_theta)};
        //-----
        x1 = vec_norm * std::cos(static_cast<floatT>(2.0) * pi<floatT> * phi);
        x2 = vec_norm * std::sin(static_cast<floatT>(2.0) * pi<floatT> * phi);
        x3 = abs_x * cos_theta;
        //-----
        // mat_su2 = SU2_comp {std::complex<floatT> (x0, x1), std::complex<floatT> (x2, x3)} * V.adjoint();
        return {SU2_comp<floatT> {std::complex<floatT> (x0, x1), std::complex<floatT> (x2, x3)} * V.adjoint()};
    // }
    // else
    // {
        // mat_su2 = SU2_comp {1.0, 0.0};
    // }
    // return mat_su2;
}

//-----
// Heatbath update for SU(3) using Cabibbo-Marinari method
// TODO: Extend to SU(N) with Ncol?

void HeatbathSU3(GaugeField& Gluon, const int n_sweep, std::uniform_real_distribution<floatT>& distribution_uniform)
{
    // For SU(2), the prefactor is 0.5 / beta
    // For SU(3), the prefactor is 0.75 / beta
    // floatT prefactor {static_cast<floatT>(0.75) / beta}; // N_c/(2 * 2)
    int N_col {3};
    floatT prefactor {static_cast<floatT>(N_col) / beta};
    for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
    for (int mu = 0; mu < 4; ++mu)
    for (int eo = 0; eo < 2; ++eo)
    {
        #pragma omp parallel for
        for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
        {
            int offset {((t + x + y) & 1) ^ eo};
            for (int z = offset; z < Nz; z+=2)
            {
                Matrix_3x3 W;
                SU2_comp<floatT> subblock;
                // Note: Our staple definition corresponds to the daggered staple in Gattringer & Lang, therefore use adjoint
                Matrix_3x3 st_adj {(WilsonAction::Staple(Gluon, {t, x, y, z}, mu)).adjoint()};
                //-----
                // Update (0, 1) subgroup
                // W = Gluon[t][x][y][z][mu] * st_adj;
                subblock = Extract01<floatT>(Gluon({t, x, y, z, mu}) * st_adj);
                Gluon({t, x, y, z, mu}) = Embed01(HeatbathSU2(subblock, prefactor, distribution_uniform)) * Gluon({t, x, y, z, mu});
                //-----
                // Update (0, 2) subgroup
                // W = Gluon[t][x][y][z][mu] * st_adj;
                subblock = Extract02<floatT>(Gluon({t, x, y, z, mu}) * st_adj);
                Gluon({t, x, y, z, mu}) = Embed02(HeatbathSU2(subblock, prefactor, distribution_uniform)) * Gluon({t, x, y, z, mu});
                //-----
                // Update (1, 2) subgroup
                // W = Gluon[t][x][y][z][mu] * st_adj;
                subblock = Extract12<floatT>(Gluon({t, x, y, z, mu}) * st_adj);
                Gluon({t, x, y, z, mu}) = Embed12(HeatbathSU2(subblock, prefactor, distribution_uniform)) * Gluon({t, x, y, z, mu});
                //-----
                // Project link to SU(3)
                SU3::Projection::GramSchmidt(Gluon({t, x, y, z, mu}));
            }
        }
    }
}

//-----
// TODO: Metadynamics

template<typename FuncT>
void MetadynamicsLocal(GaugeField& Gluon, GaugeField& Gluon1, GaugeField& Gluon2, GaugeField& Gluon3, MetaBiasPotential& Metapotential, FuncT&& CV_function, double& CV_old, const int n_sweep_heatbath, const int n_sweep_orelax, std::uniform_real_distribution<floatT>& distribution_prob, std::uniform_real_distribution<floatT>& distribution_uniform)
{
    // Copy old field so we can restore it in case the update gets rejected
    // In contrast to HMC, we expect the acceptance rates to be quite low, so always perform updates using Gluon1 instead of Gluon
    // TODO: Is that true? Better check to be sure
    // auto start_copy = std::chrono::system_clock::now();
    Gluon1 = Gluon;
    // auto end_copy = std::chrono::system_clock::now();
    // std::chrono::duration<double> copy_time = end_copy - start_copy;
    // std::cout << "Time for copy: " << copy_time.count() << std::endl;
    // Get old value of collective variable
    // TODO: Since the calculation is expensive in our case, we should try to reduce the number of CV calculations
    //       Instead of recomputing the CV, only compute the new CV and remember the old CV from last step
    // double CV_old {CV_function(Gluon1, Gluon2, rho_stout, 15)};
    // Perform update sweeps
    HeatbathSU3(Gluon1, n_sweep_heatbath, distribution_uniform);
    OverrelaxationSubgroupOld(Gluon1, n_sweep_orelax);
    // Iterator::Checkerboard(, n_sweep_heatbath);
    // Iterator::Checkerboard(OverrelaxationSubgroup, n_sweep_heatbath);
    // Get new value of collective variable
    double CV_new {CV_function(Gluon1, Gluon2, Gluon3, 10, rho_stout)};
    //-----
    // TODO: Calculate difference in metapotential
    double DeltaV {Metapotential.ReturnPotential(CV_new) - Metapotential.ReturnPotential(CV_old)};
    // Metropolis accept-reject step
    double p {std::exp(-DeltaV)};
    #if defined(_OPENMP)
    double q {distribution_prob(prng_vector[omp_get_thread_num()])};
    #else
    double q {distribution_prob(generator_rand)};
    #endif
    if (q <= p)
    {
        // std::cout << "Accepted!" << std::endl;
        Gluon = Gluon1;
        CV_old = CV_new;
        // TODO: Track acceptance rate
        // acceptance_count_metadyn += 1;
        // TODO: Update metapotential
        if constexpr(metapotential_updated)
        {
            Metapotential.UpdatePotential(CV_new);
        }
    }
    else
    {
        // std::cout << "Rejected" << std::endl;
    }
}

double MetaCharge(const GaugeField& Gluon, GaugeField& Gluon_copy1, GaugeField& Gluon_copy2, const int n_smear, const double smear_param)
{
    /*StoutSmearing4D(Gluon, Gluon1);*/
    Gluon_copy1 = Gluon;
    StoutSmearingN(Gluon_copy1, Gluon_copy2, n_smear, smear_param);
    // TODO: Probably need to rewrite StoutSmearingN so we don't have to manually keep track
    // For even n_smear, we need to use Gluon1, for odd n_smear we need to use Gluon2!
    // See description of StoutSmearingN()
    if (n_smear % 2 == 0)
    {
        return TopChargeGluonicSymm(Gluon_copy1);
    }
    else
    {
        return TopChargeGluonicSymm(Gluon_copy2);
    }
    // Old version
    // Gluon_copy = Gluon;
    // WilsonFlowForward(Gluon_copy, epsilon, n_flow);
    // return TopChargeGluonicSymm(Gluon_copy);
}

//-----
// Calculates and writes observables to logfile

// void Observables(const GaugeField& Gluon, GaugeField& Gluonchain, std::ofstream& wilsonlog, const int n_count, const int n_smear)
// {
//     vector<double> Action(n_smear + 1);
//     vector<double> WLoop2(n_smear + 1);
//     vector<double> WLoop4(n_smear + 1);
//     vector<double> WLoop8(n_smear + 1);
//     vector<double> PLoopRe(n_smear + 1);
//     vector<double> PLoopIm(n_smear + 1);
//     vector<std::complex<double>> PLoop(n_smear + 1);
//     // vector<double> TopologicalCharge(n_smear + 1);
//     vector<double> TopologicalChargeSymm(n_smear + 1);
//     vector<double> TopologicalChargeUnimproved(n_smear + 1);

//     // Unsmeared observables
//     // auto start_action = std::chrono::system_clock::now();
//     Action[0] = WilsonAction::ActionNormalized(Gluon);
//     // auto end_action = std::chrono::system_clock::now();
//     // std::chrono::duration<double> action_time = end_action - start_action;
//     // cout << "Time for calculating action: " << action_time.count() << endl;

//     // auto start_wilson = std::chrono::system_clock::now();
//     WLoop2[0] = WilsonLoop<0, 2,  true>(Gluon, Gluonchain);
//     // auto end_wilson = std::chrono::system_clock::now();
//     // std::chrono::duration<double> wilson_time = end_wilson - start_wilson;
//     // cout << "Time for calculating wilson 2: " << wilson_time.count() << endl;

//     // start_wilson = std::chrono::system_clock::now();
//     WLoop4[0] = WilsonLoop<2, 4, false>(Gluon, Gluonchain);
//     // end_wilson = std::chrono::system_clock::now();
//     // wilson_time = end_wilson - start_wilson;
//     // cout << "Time for calculating wilson 4: " << wilson_time.count() << endl;

//     // start_wilson = std::chrono::system_clock::now();
//     WLoop8[0] = WilsonLoop<4, 8, false>(Gluon, Gluonchain);
//     // end_wilson = std::chrono::system_clock::now();
//     // wilson_time = end_wilson - start_wilson;
//     // cout << "Time for calculating wilson 8: " << wilson_time.count() << endl;

//     // auto start_polyakov = std::chrono::system_clock::now();
//     PLoop[0] = PolyakovLoop(Gluon);
//     // auto end_polyakov = std::chrono::system_clock::now();
//     // std::chrono::duration<double> polyakov_time = end_polyakov - start_polyakov;
//     // cout << "Time for calculating Polyakov: " << polyakov_time.count() << endl;

//     // auto start_topcharge = std::chrono::system_clock::now();
//     // TopologicalCharge[0] = TopChargeGluonic(Gluon);
//     // auto end_topcharge = std::chrono::system_clock::now();
//     // std::chrono::duration<double> topcharge_time = end_topcharge - start_topcharge;
//     // cout << "Time for calculating topcharge: " << topcharge_time.count() << endl;
//     // auto start_topcharge_symm = std::chrono::system_clock::now();
//     TopologicalChargeSymm[0] = TopChargeGluonicSymm(Gluon);
//     // auto end_topcharge_symm = std::chrono::system_clock::now();
//     // std::chrono::duration<double> topcharge_symm_time = end_topcharge_symm - start_topcharge_symm;
//     // cout << "Time for calculating topcharge (symm): " << topcharge_symm_time.count() << endl;
//     // auto start_topcharge_plaq = std::chrono::system_clock::now();
//     TopologicalChargeUnimproved[0] = TopChargeGluonicUnimproved(Gluon);
//     // auto end_topcharge_plaq = std::chrono::system_clock::now();
//     // std::chrono::duration<double> topcharge_plaq_time = end_topcharge_plaq - start_topcharge_plaq;
//     // cout << "Time for calculating topcharge (plaq): " << topcharge_plaq_time.count() << endl;

//     //-----
//     // Begin smearing
//     if (n_smear > 0)
//     {
//         // Apply smearing
//         // auto start_smearing = std::chrono::system_clock::now();
//         StoutSmearing4D(Gluon, Gluonsmeared1, rho_stout);
//         // auto end_smearing = std::chrono::system_clock::now();
//         // std::chrono::duration<double> smearing_time = end_smearing - start_smearing;
//         // cout << "Time for calculating smearing: " << smearing_time.count() << endl;
//         // Calculate observables
//         Action[1] = WilsonAction::ActionNormalized(Gluonsmeared1);
//         WLoop2[1] = WilsonLoop<0, 2,  true>(Gluonsmeared1, Gluonchain);
//         WLoop4[1] = WilsonLoop<2, 4, false>(Gluonsmeared1, Gluonchain);
//         WLoop8[1] = WilsonLoop<4, 8, false>(Gluonsmeared1, Gluonchain);
//         PLoop[1]  = PolyakovLoop(Gluonsmeared1);
//         // TopologicalCharge[1] = TopChargeGluonic(Gluonsmeared1);
//         TopologicalChargeSymm[1] = TopChargeGluonicSymm(Gluonsmeared1);
//         TopologicalChargeUnimproved[1] = TopChargeGluonicUnimproved(Gluonsmeared1);
//     }

//     //-----
//     // Further smearing steps
//     for (int smear_count = 2; smear_count <= n_smear; ++smear_count)
//     {
//         SmearedFieldTuple SmearedFields(Gluonsmeared1, Gluonsmeared2);
//         // GaugeField& SmearedField         = Gluonsmeared1;
//         // GaugeField& PreviousSmearedField = Gluonsmeared2;
//         // Even
//         if (smear_count % 2 == 0)
//         {
//             // Apply smearing
//             // StoutSmearing4D(*Gluonsmeared1, *Gluonsmeared2, rho_stout);
//             // std::cout << "Start" << std::endl;
//             SmearedFields = StoutSmearingN(SmearedFields.Field1, SmearedFields.Field2, n_smear_skip, rho_stout);
//             // std::cout << "End" << std::endl;
//             // Calculate observables
//             Action[smear_count] = WilsonAction::ActionNormalized(SmearedFields.Field1);
//             WLoop2[smear_count] = WilsonLoop<0, 2,  true>(SmearedFields.Field1, Gluonchain);
//             WLoop4[smear_count] = WilsonLoop<2, 4, false>(SmearedFields.Field1, Gluonchain);
//             WLoop8[smear_count] = WilsonLoop<4, 8, false>(SmearedFields.Field1, Gluonchain);
//             PLoop[smear_count]  = PolyakovLoop(SmearedFields.Field1);
//             // TopologicalCharge[smear_count] = TopChargeGluonic(Gluonsmeared2);
//             TopologicalChargeSymm[smear_count] = TopChargeGluonicSymm(SmearedFields.Field1);
//             TopologicalChargeUnimproved[smear_count] = TopChargeGluonicUnimproved(SmearedFields.Field1);

//         }
//         // Odd
//         else
//         {
//             // Apply smearing
//             // StoutSmearing4D(*Gluonsmeared2, *Gluonsmeared1, rho_stout);
//             // StoutSmearingN(Gluonsmeared2, Gluonsmeared1, n_smear_skip, rho_stout);
//             // std::cout << "Start" << std::endl;
//             SmearedFields = StoutSmearingN(SmearedFields.Field1, SmearedFields.Field2, n_smear_skip, rho_stout);
//             // std::cout << "End" << std::endl;
//             // Calculate observables
//             Action[smear_count] = WilsonAction::ActionNormalized(SmearedFields.Field1);
//             WLoop2[smear_count] = WilsonLoop<0, 2,  true>(SmearedFields.Field1, Gluonchain);
//             WLoop4[smear_count] = WilsonLoop<2, 4, false>(SmearedFields.Field1, Gluonchain);
//             WLoop8[smear_count] = WilsonLoop<4, 8, false>(SmearedFields.Field1, Gluonchain);
//             PLoop[smear_count]  = PolyakovLoop(SmearedFields.Field1);
//             // TopologicalCharge[smear_count] = TopChargeGluonic(Gluonsmeared1);
//             TopologicalChargeSymm[smear_count] = TopChargeGluonicSymm(SmearedFields.Field1);
//             TopologicalChargeUnimproved[smear_count] = TopChargeGluonicUnimproved(SmearedFields.Field1);
//         }
//     }

//     //-----
//     std::transform(PLoop.begin(), PLoop.end(), PLoopRe.begin(), [](const auto& element){return std::real(element);});
//     std::transform(PLoop.begin(), PLoop.end(), PLoopIm.begin(), [](const auto& element){return std::imag(element);});

//     //-----
//     // Write to logfile
//     std::time_t log_time {std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
//     // datalog << "[Step " << n_count << "] " << std::ctime(&log_time) << "\n";
//     // datalog << "[Step " << n_count << "] -" << std::ctime(&log_time) << "-";
//     datalog << "[Step " << n_count << "] -" << std::put_time(std::localtime(&log_time), "%c") << "-\n";
//     //-----
//     if constexpr(n_hmc != 0)
//     {
//         datalog << "DeltaH: " << DeltaH << "\n";
//     }
//     //-----
//     datalog << "Wilson_Action: ";
//     // std::copy(Action.cbegin(), std::prev(Action.cend()), std::ostream_iterator<double>(datalog, " "));
//     std::copy(std::cbegin(Action), std::prev(std::cend(Action)), std::ostream_iterator<double>(datalog, " "));
//     datalog << Action.back() << "\n";
//     //-----
//     datalog << "Wilson_loop(L=2): ";
//     // std::copy(WLoop2.cbegin(), std::prev(WLoop2.cend()), std::ostream_iterator<double>(datalog, " "));
//     std::copy(std::cbegin(WLoop2), std::prev(std::cend(WLoop2)), std::ostream_iterator<double>(datalog, " "));
//     datalog << WLoop2.back() << "\n";
//     //-----
//     datalog << "Wilson_loop(L=4): ";
//     // std::copy(WLoop4.cbegin(), std::prev(WLoop4.cend()), std::ostream_iterator<double>(datalog, " "));
//     std::copy(std::cbegin(WLoop4), std::prev(std::cend(WLoop4)), std::ostream_iterator<double>(datalog, " "));
//     datalog << WLoop4.back() << "\n";
//     //-----
//     datalog << "Wilson_loop(L=8): ";
//     // std::copy(WLoop8.cbegin(), std::prev(WLoop8.cend()), std::ostream_iterator<double>(datalog, " "));
//     std::copy(std::cbegin(WLoop8), std::prev(std::cend(WLoop8)), std::ostream_iterator<double>(datalog, " "));
//     datalog << WLoop8.back() << "\n"; //<< endl;
//     //-----
//     datalog << "Polyakov_loop(Re): ";
//     std::copy(std::cbegin(PLoopRe), std::prev(std::cend(PLoopRe)), std::ostream_iterator<double>(datalog, " "));
//     datalog << PLoopRe.back() << "\n";
//     //-----
//     datalog << "Polyakov_loop(Im): ";
//     std::copy(std::cbegin(PLoopIm), std::prev(std::cend(PLoopIm)), std::ostream_iterator<double>(datalog, " "));
//     datalog << PLoopIm.back() << "\n";


//     // datalog << "TopChargeClov: ";
//     // std::copy(std::cbegin(TopologicalCharge), std::prev(std::cend(TopologicalCharge)), std::ostream_iterator<double>(datalog, " "));
//     // datalog << TopologicalCharge.back() << "\n"; //<< endl;

//     datalog << "TopChargeClov: ";
//     std::copy(std::cbegin(TopologicalChargeSymm), std::prev(std::cend(TopologicalChargeSymm)), std::ostream_iterator<double>(datalog, " "));
//     datalog << TopologicalChargeSymm.back() << "\n"; //<< endl;

//     datalog << "TopChargePlaq: ";
//     std::copy(std::cbegin(TopologicalChargeUnimproved), std::prev(std::cend(TopologicalChargeUnimproved)), std::ostream_iterator<double>(datalog, " "));
//     datalog << TopologicalChargeUnimproved.back() << "\n" << endl;
// }

void Observables(const GaugeField& Gluon, GaugeField& Gluonchain, std::ofstream& wilsonlog, const int n_count, const int n_smear)
{
    vector<double> Action(n_smear + 1);
    vector<double> WLoop2(n_smear + 1);
    vector<double> WLoop4(n_smear + 1);
    vector<double> WLoop8(n_smear + 1);
    vector<double> PLoopRe(n_smear + 1);
    vector<double> PLoopIm(n_smear + 1);
    vector<std::complex<double>> PLoop(n_smear + 1);
    // vector<double> TopologicalCharge(n_smear + 1);
    vector<double> TopologicalChargeSymm(n_smear + 1);
    vector<double> TopologicalChargeUnimproved(n_smear + 1);

    // Unsmeared observables
    // auto start_action = std::chrono::system_clock::now();
    Action[0] = WilsonAction::ActionNormalized(Gluon);
    // auto end_action = std::chrono::system_clock::now();
    // std::chrono::duration<double> action_time = end_action - start_action;
    // cout << "Time for calculating action: " << action_time.count() << endl;

    // auto start_wilson = std::chrono::system_clock::now();
    WLoop2[0] = WilsonLoop<0, 2,  true>(Gluon, Gluonchain);
    // auto end_wilson = std::chrono::system_clock::now();
    // std::chrono::duration<double> wilson_time = end_wilson - start_wilson;
    // cout << "Time for calculating wilson 2: " << wilson_time.count() << endl;

    // start_wilson = std::chrono::system_clock::now();
    WLoop4[0] = WilsonLoop<2, 4, false>(Gluon, Gluonchain);
    // end_wilson = std::chrono::system_clock::now();
    // wilson_time = end_wilson - start_wilson;
    // cout << "Time for calculating wilson 4: " << wilson_time.count() << endl;

    // start_wilson = std::chrono::system_clock::now();
    WLoop8[0] = WilsonLoop<4, 8, false>(Gluon, Gluonchain);
    // end_wilson = std::chrono::system_clock::now();
    // wilson_time = end_wilson - start_wilson;
    // cout << "Time for calculating wilson 8: " << wilson_time.count() << endl;

    // auto start_polyakov = std::chrono::system_clock::now();
    PLoop[0] = PolyakovLoop(Gluon);
    // auto end_polyakov = std::chrono::system_clock::now();
    // std::chrono::duration<double> polyakov_time = end_polyakov - start_polyakov;
    // cout << "Time for calculating Polyakov: " << polyakov_time.count() << endl;

    // auto start_topcharge = std::chrono::system_clock::now();
    // TopologicalCharge[0] = TopChargeGluonic(Gluon);
    // auto end_topcharge = std::chrono::system_clock::now();
    // std::chrono::duration<double> topcharge_time = end_topcharge - start_topcharge;
    // cout << "Time for calculating topcharge: " << topcharge_time.count() << endl;
    // auto start_topcharge_symm = std::chrono::system_clock::now();
    TopologicalChargeSymm[0] = TopChargeGluonicSymm(Gluon);
    // auto end_topcharge_symm = std::chrono::system_clock::now();
    // std::chrono::duration<double> topcharge_symm_time = end_topcharge_symm - start_topcharge_symm;
    // cout << "Time for calculating topcharge (symm): " << topcharge_symm_time.count() << endl;
    // auto start_topcharge_plaq = std::chrono::system_clock::now();
    TopologicalChargeUnimproved[0] = TopChargeGluonicUnimproved(Gluon);
    // auto end_topcharge_plaq = std::chrono::system_clock::now();
    // std::chrono::duration<double> topcharge_plaq_time = end_topcharge_plaq - start_topcharge_plaq;
    // cout << "Time for calculating topcharge (plaq): " << topcharge_plaq_time.count() << endl;

    //-----
    // Begin smearing
    if (n_smear > 0)
    {
        // Apply smearing
        // auto start_smearing = std::chrono::system_clock::now();
        StoutSmearing4D(Gluon, Gluonsmeared1, rho_stout);
        // auto end_smearing = std::chrono::system_clock::now();
        // std::chrono::duration<double> smearing_time = end_smearing - start_smearing;
        // cout << "Time for calculating smearing: " << smearing_time.count() << endl;
        // Calculate observables
        Action[1] = WilsonAction::ActionNormalized(Gluonsmeared1);
        WLoop2[1] = WilsonLoop<0, 2,  true>(Gluonsmeared1, Gluonchain);
        WLoop4[1] = WilsonLoop<2, 4, false>(Gluonsmeared1, Gluonchain);
        WLoop8[1] = WilsonLoop<4, 8, false>(Gluonsmeared1, Gluonchain);
        PLoop[1]  = PolyakovLoop(Gluonsmeared1);
        // TopologicalCharge[1] = TopChargeGluonic(Gluonsmeared1);
        TopologicalChargeSymm[1] = TopChargeGluonicSymm(Gluonsmeared1);
        TopologicalChargeUnimproved[1] = TopChargeGluonicUnimproved(Gluonsmeared1);
    }

    //-----
    // Further smearing steps
    for (int smear_count = 2; smear_count <= n_smear; ++smear_count)
    {
        // Even
        if (smear_count % 2 == 0)
        {
            // Apply smearing
            // StoutSmearing4D(*Gluonsmeared1, *Gluonsmeared2, rho_stout);
            StoutSmearingN(Gluonsmeared1, Gluonsmeared2, n_smear_skip, rho_stout);
            // TODO: FIX THIS, INCORRECT IF n_smear_skip is even!
            // if (n_smear_skip % 2 == 0)
            // {
            //     // placeholder
            // }
            // else
            // {
            //     // placeholder
            // }
            // Calculate observables
            Action[smear_count] = WilsonAction::ActionNormalized(Gluonsmeared2);
            WLoop2[smear_count] = WilsonLoop<0, 2,  true>(Gluonsmeared2, Gluonchain);
            WLoop4[smear_count] = WilsonLoop<2, 4, false>(Gluonsmeared2, Gluonchain);
            WLoop8[smear_count] = WilsonLoop<4, 8, false>(Gluonsmeared2, Gluonchain);
            PLoop[smear_count]  = PolyakovLoop(Gluonsmeared2);
            // TopologicalCharge[smear_count] = TopChargeGluonic(Gluonsmeared2);
            TopologicalChargeSymm[smear_count] = TopChargeGluonicSymm(Gluonsmeared2);
            TopologicalChargeUnimproved[smear_count] = TopChargeGluonicUnimproved(Gluonsmeared2);
        }
        // Odd
        else
        {
            // Apply smearing
            // StoutSmearing4D(*Gluonsmeared2, *Gluonsmeared1, rho_stout);
            StoutSmearingN(Gluonsmeared2, Gluonsmeared1, n_smear_skip, rho_stout);
            // Calculate observables
            Action[smear_count] = WilsonAction::ActionNormalized(Gluonsmeared1);
            WLoop2[smear_count] = WilsonLoop<0, 2,  true>(Gluonsmeared1, Gluonchain);
            WLoop4[smear_count] = WilsonLoop<2, 4, false>(Gluonsmeared1, Gluonchain);
            WLoop8[smear_count] = WilsonLoop<4, 8, false>(Gluonsmeared1, Gluonchain);
            PLoop[smear_count]  = PolyakovLoop(Gluonsmeared1);
            // TopologicalCharge[smear_count] = TopChargeGluonic(Gluonsmeared1);
            TopologicalChargeSymm[smear_count] = TopChargeGluonicSymm(Gluonsmeared1);
            TopologicalChargeUnimproved[smear_count] = TopChargeGluonicUnimproved(Gluonsmeared1);
        }
    }

    //-----
    std::transform(PLoop.begin(), PLoop.end(), PLoopRe.begin(), [](const auto& element){return std::real(element);});
    std::transform(PLoop.begin(), PLoop.end(), PLoopIm.begin(), [](const auto& element){return std::imag(element);});

    //-----
    // Write to logfile
    std::time_t log_time {std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
    // datalog << "[Step " << n_count << "] " << std::ctime(&log_time) << "\n";
    // datalog << "[Step " << n_count << "] -" << std::ctime(&log_time) << "-";
    datalog << "[Step " << n_count << "] -" << std::put_time(std::localtime(&log_time), "%c") << "-\n";
    //-----
    if constexpr(n_hmc != 0)
    {
        datalog << "DeltaH: " << DeltaH << "\n";
    }
    //-----
    datalog << "Wilson_Action: ";
    // std::copy(Action.cbegin(), std::prev(Action.cend()), std::ostream_iterator<double>(datalog, " "));
    std::copy(std::cbegin(Action), std::prev(std::cend(Action)), std::ostream_iterator<double>(datalog, " "));
    datalog << Action.back() << "\n";
    //-----
    datalog << "Wilson_loop(L=2): ";
    // std::copy(WLoop2.cbegin(), std::prev(WLoop2.cend()), std::ostream_iterator<double>(datalog, " "));
    std::copy(std::cbegin(WLoop2), std::prev(std::cend(WLoop2)), std::ostream_iterator<double>(datalog, " "));
    datalog << WLoop2.back() << "\n";
    //-----
    datalog << "Wilson_loop(L=4): ";
    // std::copy(WLoop4.cbegin(), std::prev(WLoop4.cend()), std::ostream_iterator<double>(datalog, " "));
    std::copy(std::cbegin(WLoop4), std::prev(std::cend(WLoop4)), std::ostream_iterator<double>(datalog, " "));
    datalog << WLoop4.back() << "\n";
    //-----
    datalog << "Wilson_loop(L=8): ";
    // std::copy(WLoop8.cbegin(), std::prev(WLoop8.cend()), std::ostream_iterator<double>(datalog, " "));
    std::copy(std::cbegin(WLoop8), std::prev(std::cend(WLoop8)), std::ostream_iterator<double>(datalog, " "));
    datalog << WLoop8.back() << "\n"; //<< endl;
    //-----
    datalog << "Polyakov_loop(Re): ";
    std::copy(std::cbegin(PLoopRe), std::prev(std::cend(PLoopRe)), std::ostream_iterator<double>(datalog, " "));
    datalog << PLoopRe.back() << "\n";
    //-----
    datalog << "Polyakov_loop(Im): ";
    std::copy(std::cbegin(PLoopIm), std::prev(std::cend(PLoopIm)), std::ostream_iterator<double>(datalog, " "));
    datalog << PLoopIm.back() << "\n";


    // datalog << "TopChargeClov: ";
    // std::copy(std::cbegin(TopologicalCharge), std::prev(std::cend(TopologicalCharge)), std::ostream_iterator<double>(datalog, " "));
    // datalog << TopologicalCharge.back() << "\n"; //<< endl;

    datalog << "TopChargeClov: ";
    std::copy(std::cbegin(TopologicalChargeSymm), std::prev(std::cend(TopologicalChargeSymm)), std::ostream_iterator<double>(datalog, " "));
    datalog << TopologicalChargeSymm.back() << "\n"; //<< endl;

    datalog << "TopChargePlaq: ";
    std::copy(std::cbegin(TopologicalChargeUnimproved), std::prev(std::cend(TopologicalChargeUnimproved)), std::ostream_iterator<double>(datalog, " "));
    datalog << TopologicalChargeUnimproved.back() << "\n" << endl;
}

//-----

int main()
{
    // iostream not synchronized with corresponding C streams, might cause a problem with C libraries and might not be thread safe
    std::ios_base::sync_with_stdio(false);
    cout << std::setprecision(12) << std::fixed;
    datalog << std::setprecision(12) << std::fixed;

    Configuration();
    prng_vector = CreatePRNGs();
    if constexpr(n_hmc != 0)
    {
        ndist_vector = CreateNormal();
    }

    // acceptance_count = 0;
    // acceptance_count_or = 0;
    // Default width of random numbers used in Metropolis update is 0.5
    // floatT epsilon {0.001}; // Used to test stability of instanton configurations
    floatT epsilon {0.5};

    std::uniform_real_distribution<floatT> distribution_prob(0.0, 1.0);
    std::uniform_real_distribution<floatT> distribution_uniform(0.0, 1.0);
    std::uniform_int_distribution<int> distribution_choice(1, 8);

    CreateFiles();
    SetGluonToOne(Gluon);

    auto startcalc {std::chrono::system_clock::now()};
    datalog.open(logfilepath, std::fstream::out | std::fstream::app);
    wilsonlog.open(wilsonfilepath, std::fstream::out | std::fstream::app);

    // Instanton multiplication test
    // std::uniform_real_distribution<floatT> distribution_unitary(-epsilon, epsilon);
    // InstantonStart(*Gluon, 2);
    // for (int n_count = 0; n_count < 20; ++n_count)
    // {
    //     // MetropolisUpdate(*Gluon, n_metro, acceptance_count, epsilon, distribution_prob, distribution_choice, distribution_unitary);
    //     HeatbathSU3(*Gluon, 1, distribution_uniform);
    // }

    // Observables(*Gluon, *Gluonchain, wilsonlog, 0, n_smear);
    // // MultiplyInstanton(*Gluon, 1);
    // MultiplyLocalInstanton(*Gluon);
    // Observables(*Gluon, *Gluonchain, wilsonlog, 1, n_smear);
    // std::exit(0);
    // Instanton start test
    // for (int Q = 0; Q < 10; ++Q)
    // {
    //     InstantonStart(*Gluon, Q);
    //     Observables(*Gluon, *Gluonchain, wilsonlog, Q, n_smear);
    //     cout << "This configuration should have charge " << Q <<". The charge is: " << TopChargeGluonic(*Gluon) << endl;
    //     cout << "This configuration should have charge " << Q <<". The charge is: " << TopChargeGluonicUnimproved(*Gluon) << endl;
    //     MultiplyInstanton(*Gluon, 1);
    //     Observables(*Gluon, *Gluonchain, wilsonlog, Q, n_smear);
    //     cout << "After multiplication the charge is: " << TopChargeGluonic(*Gluon) << endl;
    //     cout << "After multiplication the charge is: " << TopChargeGluonicUnimproved(*Gluon) << endl;
    //     // InstantonStart(*Gluon, -Q);
    //     // cout << "This configuration should have charge " << -Q <<". The charge is: " << TopChargeGluonic(*Gluon) << endl;
    //     // cout << "This configuration should have charge " << -Q <<". The charge is: " << TopChargeGluonicUnimproved(*Gluon) << endl;
    // }

    // Generate 1 instanton
    // LocalInstantonStart(*Gluon);
    // Observables(*Gluon, *Gluonchain, wilsonlog, 0, n_smear);

    // InstantonStart(*Gluon, 1);
    // Observables(*Gluon, *Gluonchain, wilsonlog, 0, n_smear);
    // for (int n_count = 0; n_count < 20; ++n_count)
    // {
    //     MetropolisUpdate(*Gluon, n_metro, acceptance_count, epsilon, distribution_prob, distribution_choice, distribution_unitary);
    // }
    // Observables(*Gluon, *Gluonchain, wilsonlog, -1, 0);
    // for (int flow_count = 0; flow_count < 10; ++flow_count)
    // {
    //     WilsonFlowForward(*Gluon, 0.06, 1);
    //     Observables(*Gluon, *Gluonchain, wilsonlog, flow_count, 0);
    // }
    // // MultiplyInstanton(*Gluon, 1);
    // MultiplyLocalInstanton(*Gluon);
    // Observables(*Gluon, *Gluonchain, wilsonlog, -1, 0);
    // for (int flow_count = 0; flow_count < 10; ++flow_count)
    // {
    //     WilsonFlowBackward(*Gluon, *Gluonsmeared1, -0.06, 1);
    //     Observables(*Gluon, *Gluonchain, wilsonlog, flow_count, 0);
    // }
    // std::exit(0);

    // auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - startcalc;
    // std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    // PrintFinal(datalog, acceptance_count, acceptance_count_or, epsilon, end_time, elapsed_seconds);
    // datalog.close();
    // datalog.clear();

    // datalog.open(parameterfilepath, std::fstream::out | std::fstream::app);
    // PrintFinal(datalog, acceptance_count, acceptance_count_or, epsilon, end_time, elapsed_seconds);
    // datalog.close();
    // datalog.clear();

    // PrintFinal(wilsonlog, acceptance_count, acceptance_count_or, epsilon, end_time, elapsed_seconds);
    // wilsonlog.close();
    // wilsonlog.clear();

    // std::exit(0);

    // MetropolisUpdate(*Gluon, n_metro, acceptance_count, epsilon, distribution_prob, distribution_choice, distribution_unitary);
    // Observables(*Gluon, *Gluonchain, wilsonlog, 0, n_smear);
    // // Multiply with 1 instanton, resulting configuration should have charge (Q_1 + 1) * (1 + 1) = (1 + 1) * (1 + 1) = 4
    // InstantonStart(*Gluon, 1);
    // MultiplyInstanton(*Gluon, 1);
    // Observables(*Gluon, *Gluonchain, wilsonlog, 1, n_smear);

    // MetropolisUpdate(*Gluon, n_metro, acceptance_count, epsilon, distribution_prob, distribution_choice, distribution_unitary);
    // Observables(*Gluon, *Gluonchain, wilsonlog, 1, n_smear);
    // // Generate 4 instanton
    // InstantonStart(*Gluon, 4);
    // Observables(*Gluon, *Gluonchain, wilsonlog, 2, n_smear);

    // MetropolisUpdate(*Gluon, n_metro, acceptance_count, epsilon, distribution_prob, distribution_choice, distribution_unitary);
    // Observables(*Gluon, *Gluonchain, wilsonlog, 2, n_smear);

    // TODO: Rewrite this, maybe keep metadynamics updates in separate main?
    // if constexpr(metadynamics_enabled)
    // {
        // CV_min, CV_max, bin_number, weight, threshold_weight
        MetaBiasPotential TopBiasPotential{-8, 8, 800, 0.02, 1000.0};
        TopBiasPotential.SaveMetaParameters(metapotentialfilepath);
        // Calculate first CV so that we don't have to recompute it later on
        double CV {0.0};
        if constexpr(metadynamics_enabled)
        {
            CV = MetaCharge(Gluon, Gluonsmeared1, Gluonsmeared2, 10, rho_stout);
        }
        // auto CV_function = [](){MetaCharge(*Gluon, *Gluonsmeared1, *Gluonsmeared2, 15);};
    // }

    // Initialize update functors
    HeatbathKernel               Heatbath(Gluon, distribution_uniform);
    // OverrelaxationDirectKernel   OverrelaxationDirect(Gluon, distribution_prob);
    OverrelaxationSubgroupKernel OverrelaxationSubgroup(Gluon);

    // When using HMC, the thermalization is done without accept-reject step
    if constexpr(n_hmc != 0)
    {
        datalog << "[HMC start thermalization]\n";
        for (int n_count = 0; n_count < 20; ++n_count)
        {
            // std::cout << "HMC accept/reject (therm): " << HMC::HMCGauge(*Gluon, *Gluonsmeared1, *Gluonsmeared2, HMC::Leapfrog, 100, false, distribution_prob) << std::endl;
            HMC::HMCGauge(Gluon, Gluonsmeared1, Gluonsmeared2, acceptance_count_hmc, HMC::OMF_4, 10, false, distribution_prob);
        }
        datalog << "[HMC end thermalization]\n" << std::endl;
    }

    for (int n_count = 0; n_count < n_run; ++n_count)
    {
        // InstantonStart(*Gluon, 2);
        // auto start_update_metro {std::chrono::system_clock::now()};
        if constexpr(n_metro != 0 && multi_hit != 0)
        {
            std::uniform_real_distribution<floatT> distribution_unitary(-epsilon, epsilon);
            MetropolisUpdate(Gluon, n_metro, acceptance_count, epsilon, distribution_prob, distribution_choice, distribution_unitary);
        }
        // auto end_update_metro {std::chrono::system_clock::now()};
        // std::chrono::duration<double> update_time_metro {end_update_metro - start_update_metro};
        // cout << "Time for " << n_metro << " Metropolis updates: " << update_time_metro.count() << endl;
        //-----
        // auto start_update_heatbath {std::chrono::system_clock::now()};
        if constexpr(n_heatbath != 0)
        {
            // HeatbathSU3(Gluon, n_heatbath, distribution_uniform);
            Iterator::Checkerboard(Heatbath, n_heatbath);
        }
        // auto end_update_heatbath {std::chrono::system_clock::now()};
        // std::chrono::duration<double> update_time_heatbath {end_update_heatbath - start_update_heatbath};
        // cout << "Time for " << n_heatbath << " heatbath updates: " << update_time_heatbath.count() << endl;
        //-----
        // auto start_update_hmc {std::chrono::system_clock::now()};
        if constexpr(n_hmc != 0)
        {
            // std::cout << "HMC accept/reject: " << HMC::HMCGauge(*Gluon, *Gluonsmeared1, *Gluonsmeared2, HMC::OMF_4, n_hmc, true, distribution_prob) << std::endl;
            HMC::HMCGauge(Gluon, Gluonsmeared1, Gluonsmeared2, acceptance_count_hmc, HMC::OMF_4, n_hmc, true, distribution_prob);
        }
        // auto end_update_hmc {std::chrono::system_clock::now()};
        // std::chrono::duration<double> update_time_hmc {end_update_hmc - start_update_hmc};
        // cout << "Time for one HMC trajectory: " << update_time_hmc.count() << endl;
        //-----
        // auto start_update_or = std::chrono::system_clock::now();
        if constexpr(n_orelax != 0)
        {
            // Iterator::CheckerboardSum(OverrelaxationDirect, acceptance_count_or, n_orelax);
            Iterator::Checkerboard(OverrelaxationSubgroup, n_orelax);
        }
        // auto end_update_or = std::chrono::system_clock::now();
        // std::chrono::duration<double> update_time_or {end_update_or - start_update_or};
        // cout << "Time for " << n_orelax << " OR updates: " << update_time_or.count() << endl;
        //-----
        // auto start_update_meta = std::chrono::system_clock::now();
        if constexpr(metadynamics_enabled)
        {
            MetadynamicsLocal(Gluon, Gluonsmeared1, Gluonsmeared2, Gluonsmeared3, TopBiasPotential, MetaCharge, CV, 1, 4, distribution_prob, distribution_uniform);
        }
        // auto end_update_meta = std::chrono::system_clock::now();
        // std::chrono::duration<double> update_time_meta {end_update_meta - start_update_meta};
        // cout << "Time for meta update: " << update_time_meta.count() << endl;
        if (n_count % expectation_period == 0)
        {
            // auto start_observable = std::chrono::system_clock::now();
            Observables(Gluon, Gluonchain, wilsonlog, n_count, n_smear);
            // auto end_observable = std::chrono::system_clock::now();
            // std::chrono::duration<double> observable_time {end_observable - start_observable};
            // cout << "Time for calculating observables: " << observable_time.count() << endl;
            if constexpr(metadynamics_enabled)
            {
                TopBiasPotential.SaveMetaPotential(metapotentialfilepath);
            }
        }
    }

    auto end {std::chrono::system_clock::now()};
    std::chrono::duration<double> elapsed_seconds {end - startcalc};
    std::time_t end_time {std::chrono::system_clock::to_time_t(end)};

    //-----
    // Print acceptance rates, PRNG width, and required time to terminal and to files

    cout << "\n";
    PrintFinal(cout, acceptance_count, acceptance_count_or, acceptance_count_hmc, epsilon, end_time, elapsed_seconds);

    PrintFinal(datalog, acceptance_count, acceptance_count_or, acceptance_count_hmc, epsilon, end_time, elapsed_seconds);
    datalog.close();
    datalog.clear();

    datalog.open(parameterfilepath, std::fstream::out | std::fstream::app);
    PrintFinal(datalog, acceptance_count, acceptance_count_or, acceptance_count_hmc, epsilon, end_time, elapsed_seconds);
    datalog.close();
    datalog.clear();

    PrintFinal(wilsonlog, acceptance_count, acceptance_count_or, acceptance_count_hmc, epsilon, end_time, elapsed_seconds);
    wilsonlog.close();
    wilsonlog.clear();
}
