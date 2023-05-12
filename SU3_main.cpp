// Pure SU(3) theory
// Debug Flags: -DDEBUG_MODE_TERMINAL, -DFIXED_SEED

// #define EIGEN_USE_MKL_ALL

// Non-standard library headers
#include "LettuceGauge/actions/gauge/rectangular_action.hpp"
#include "LettuceGauge/coords.hpp"
#include "LettuceGauge/defines.hpp"
#include "LettuceGauge/IO/ansi_colors.hpp"
#include "LettuceGauge/IO/config_io/bmw_format.hpp"
#include "LettuceGauge/IO/config_io/bridge_text_format.hpp"
#include "LettuceGauge/iterators/iterators.hpp"
#include "LettuceGauge/lattice.hpp"
#include "LettuceGauge/math/su2.hpp"
#include "LettuceGauge/math/su3.hpp"
#include "LettuceGauge/math/su3_exp.hpp"
#include "LettuceGauge/metadynamics.hpp"
#include "LettuceGauge/observables/observables.hpp"
#include "LettuceGauge/observables/clover.hpp"
#include "LettuceGauge/observables/plaquette.hpp"
#include "LettuceGauge/observables/field_strength_tensor.hpp"
#include "LettuceGauge/observables/polyakov_loop.hpp"
#include "LettuceGauge/observables/topological_charge.hpp"
#include "LettuceGauge/observables/wilson_loop.hpp"
#include "LettuceGauge/smearing/cooling.hpp"
#include "LettuceGauge/smearing/stout_smearing.hpp"
#include "LettuceGauge/smearing/gradient_flow.hpp"
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
FullTensor                   F_tensor;
// std::unique_ptr<Full_tensor> F_tensor      {std::make_unique<Full_tensor>()};
// std::unique_ptr<Full_tensor> Q_tensor      {std::make_unique<Full_tensor>()};

//-----
// Overload << for vectors and arrays?

// template<typename T>
// std::ostream& operator<<(ostream& out, const std::vector<T>& container)
// {
//     out << "Container dump begins: ";
//     std::copy(container.cbegin(), container.cend(), std::ostream_iterator<T>(out, " "));
//     out << "\n";
//     return out;
// }

// template<typename T>
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
// TODO: Should probably make clear that this version works for the terminal only
//       Rename this to ValidatedInTerminal and write alternative version for reading from files?

template<typename T>
void ValidatedIn(const std::string& message, T& target)
{
    // Keep count of tries and abort after too many tries (e.g. important when using nohup)
    size_t count {0};
    while (std::cout << Lettuce::Color::BoldBlue << message << Lettuce::Color::Reset << "\n" && !(std::cin >> target) && count < 10)
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << Lettuce::Color::Red << "Invalid input." << Lettuce::Color::Reset << "\n";
        ++count;
    }
}

//-----
// Receives simulation parameters from user input
// TODO: We should probably wrap all parameters in a struct and also write a single print function we can reuse for both printing
//       to the terminal and writing to files.

void Configuration()
{
    std::cout << Lettuce::Color::BoldBlue << "\n\n+------------------------------------------------+\n";
    std::cout << std::left << std::setw(49) << "| SU(3) theory simulation" << "|\n";
    std::cout << std::left << std::setw(49) << "| Current version: " + program_version << "|\n";
    std::cout << "+------------------------------------------------+\n\n" << Lettuce::Color::Reset;
    // Get simulation parameters from user input
    ValidatedIn("Please enter beta: ", beta);
    ValidatedIn("Please enter n_run: ", n_run);
    ValidatedIn("Please enter expectation_period: ", expectation_period);
    n_run_inverse = 1.0 / static_cast<double>(n_run);
    if (n_metro != 0 && multi_hit != 0)
    {
        metro_norm = 1.0 / (Nt * Nx * Ny * Nz * 4.0 * n_metro * multi_hit);
    }
    std::cout << "\n" << "Gauge field precision: " << typeid(floatT).name() << "\n";
    std::cout << "beta is "                        << beta << ".\n";
    std::cout << "n_run is "                       << n_run << " and expectation_period is " << expectation_period << ".\n";
    std::cout << "n_metro is "                     << n_metro << ".\n";
    std::cout << "multi_hit is "                   << multi_hit << ".\n";
    std::cout << "metro_target_acceptance is "     << metro_target_acceptance << ".\n";
    std::cout << "n_heatbath is "                  << n_heatbath << ".\n";
    std::cout << "n_hmc is "                       << n_hmc << ".\n";
    std::cout << "n_orelax is "                    << n_orelax << ".\n";
    std::cout << "n_instanton_update is "          << n_instanton_update << ".\n";
    std::cout << "metadynamics_enabled is "        << metadynamics_enabled << ".\n";
    std::cout << "metapotential_updated is "       << metapotential_updated << ".\n";
    std::cout << "tempering_enabled is "           << tempering_enabled      << ".\n";
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
    #ifdef FIXED_SEED
    datalog << "FIXED_SEED\n";
    #endif
    datalog << starttimestring << "\n";
    datalog << "START_PARAMS\n";
    datalog << "Gauge field precision = "            << typeid(floatT).name()            << "\n";
    datalog << "Nt = "                               << Nt                               << "\n";
    datalog << "Nx = "                               << Nx                               << "\n";
    datalog << "Ny = "                               << Ny                               << "\n";
    datalog << "Nz = "                               << Nz                               << "\n";
    datalog << "beta = "                             << beta                             << "\n";
    datalog << "n_run = "                            << n_run                            << "\n";
    datalog << "expectation_period = "               << expectation_period               << "\n";
    datalog << "n_smear = "                          << n_smear                          << "\n";
    datalog << "n_smear_skip = "                     << n_smear_skip                     << "\n";
    datalog << "rho_stout = "                        << rho_stout                        << "\n";
    datalog << "n_metro = "                          << n_metro                          << "\n";
    datalog << "multi_hit = "                        << multi_hit                        << "\n";
    datalog << "metro_target_acceptance = "          << metro_target_acceptance          << "\n";
    datalog << "n_heatbath = "                       << n_heatbath                       << "\n";
    datalog << "n_hmc = "                            << n_hmc                            << "\n";
    datalog << "n_orelax = "                         << n_orelax                         << "\n";
    datalog << "n_instanton_update = "               << n_instanton_update               << "\n";
    datalog << "metadynamics_enabled = "             << metadynamics_enabled             << "\n";
    datalog << "metapotential_updated = "            << metapotential_updated            << "\n";
    datalog << "n_smear_meta = "                     << n_smear_meta                     << "\n";
    datalog << "tempering_enabled = "                << tempering_enabled                << "\n";
    datalog << "tempering_nonmetadynamics_sweeps = " << tempering_nonmetadynamics_sweeps << "\n";
    datalog << "tempering_swap_period = "            << tempering_swap_period            << "\n";
    datalog << "END_PARAMS\n"                        << std::endl;
    datalog.close();
    datalog.clear();
}

//-----
// Creates directories and files to store data

void CreateFiles()
{
    std::string LatticeSizeString    {std::to_string(Nx) + "x" + std::to_string(Ny) + "x" + std::to_string(Nz) + "x" + std::to_string(Nt)};
    std::string betaString           {std::to_string(beta)};
    std::string directoryname_prefix {"SU(3)_N=" + LatticeSizeString + "_beta=" + betaString};
    directoryname = directoryname_prefix;
    int append = 1;
    std::string appendString;
    while (std::filesystem::exists(directoryname))
    {
        appendString  = std::to_string(append);
        directoryname = directoryname_prefix + " (" + appendString + ")";
        ++append;
    }
    std::filesystem::create_directory(directoryname);
    std::cout << "\n\n" << "Created directory \"" << directoryname << "\".\n";
    logfilepath           = directoryname + "/log.txt";
    parameterfilepath     = directoryname + "/parameters.txt";
    wilsonfilepath        = directoryname + "/wilson.txt";
    metapotentialfilepath = directoryname + "/metapotential.txt";
    std::cout << Lettuce::Color::BoldBlue << "Filepath (log):\t\t"      << logfilepath                                    << "\n";
    std::cout                             << "Filepath (parameters):\t" << parameterfilepath                              << "\n";
    std::cout                             << "Filepath (wilson):\t"     << wilsonfilepath                                 << "\n";
    std::cout                             << "Filepath (metadyn):\t"    << metapotentialfilepath << Lettuce::Color::Reset << "\n";
    #ifdef DEBUG_MODE_TERMINAL
    std::cout << "DEBUG_MODE_TERMINAL\n\n";
    #endif

    //-----
    // Writes parameters to files

    std::time_t start_time        {std::chrono::system_clock::to_time_t(start)};
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
void PrintFinal(std::ostream& log, const uint_fast64_t acceptance_count, const uint_fast64_t acceptance_count_or, const uint_fast64_t acceptance_count_hmc, const uint_fast64_t acceptance_count_metadynamics_hmc, const uint_fast64_t acceptance_count_tempering, const floatT epsilon, const std::time_t& end_time, const std::chrono::duration<double>& elapsed_seconds)
{
    double or_norm        {1.0};
    if constexpr(n_orelax != 0)
    {
        or_norm        = 1.0 / (Nt * Nx * Ny * Nz * 4.0 * n_run * n_orelax);
    }
    double hmc_norm       {1.0};
    if constexpr(n_hmc != 0)
    {
        hmc_norm       = 1.0 / n_run;
    }
    double tempering_norm {1.0};
    if constexpr(tempering_enabled)
    {
        // Need to cast either numerator or denominator to floating type for the divison to work as we want
        tempering_norm = static_cast<double>(tempering_swap_period) / n_run;
    }
    double instanton_norm {1.0};
    if constexpr(n_instanton_update != 0)
    {
        instanton_norm = 1.0 / (n_run * n_instanton_update);
    }
    log << "Metro target acceptance: " << metro_target_acceptance                      << "\n";
    log << "Metro acceptance: "        << acceptance_count * metro_norm                << "\n";
    log << "OR acceptance: "           << acceptance_count_or * or_norm                << "\n";
    log << "HMC acceptance: "          << acceptance_count_hmc * hmc_norm              << "\n";
    log << "MetaD-HMC acceptance: "    << acceptance_count_metadynamics_hmc * hmc_norm << "\n";
    log << "Tempering acceptance: "    << acceptance_count_tempering * tempering_norm  << "\n";
    log << "Instanton acceptance: "    << acceptance_count_instanton * instanton_norm  << "\n";
    log << "epsilon: "                 << epsilon                                      << "\n";
    log << std::ctime(&end_time)                                                       << "\n";
    log << "Required time: "           << elapsed_seconds.count()                      << "s\n";
}

[[nodiscard]]
vector<pcg64> CreatePRNGs(const int thread_num = 0)
{
    vector<pcg64> tmp_vec;
    #if defined(_OPENMP)
        int max_thread_num {omp_get_max_threads()};
    #else
        int max_thread_num {1};
    #endif
    std::cout << "Maximum number of threads: " << max_thread_num << std::endl;
    #if defined(_OPENMP)
        if (thread_num != 0)
        {
            max_thread_num = thread_num;
            omp_set_num_threads(thread_num);
        }
    #endif
    if (max_thread_num != 1)
    {
        std::cout << "Creating PRNG vector with " << max_thread_num << " PRNGs.\n" << std::endl;
    }
    else
    {
        std::cout << "Creating PRNG vector with " << max_thread_num << " PRNG.\n" << std::endl;
    }
    for (int thread_count = 0; thread_count < max_thread_num; ++thread_count)
    {
        #ifdef FIXED_SEED
        pcg64 generator_rand_tmp(thread_count);
        tmp_vec.emplace_back(generator_rand_tmp);
        // tmp_vec.emplace_back(generator_rand_tmp(thread_count));
        #else
        pcg_extras::seed_seq_from<std::random_device> seed_source_tmp;
        pcg64 generator_rand_tmp(seed_source_tmp);
        tmp_vec.emplace_back(generator_rand_tmp);
        // tmp_vec.emplace_back(generator_rand_tmp(seed_source_tmp));
        #endif
    }
    return tmp_vec;
}

//-----
// Create vector of normal_distribution generators with mean 0 and standard deviation 1 for HMC

[[nodiscard]]
vector<std::normal_distribution<floatT>> CreateNormalDistributions(const int thread_num = 0)
{
    vector<std::normal_distribution<floatT>> tmp_vec;
    #if defined(_OPENMP)
        int max_thread_num {omp_get_max_threads()};
    #else
        int max_thread_num {1};
    #endif
    std::cout << "Maximum number of threads: " << max_thread_num << std::endl;
    #if defined(_OPENMP)
        if (thread_num != 0)
        {
            max_thread_num = thread_num;
            omp_set_num_threads(thread_num);
        }
    #endif
    if (max_thread_num != 1)
    {
        std::cout << "Creating vector of normal_distributions with " << max_thread_num << " normal_distributions.\n" << std::endl;
    }
    else
    {
        std::cout << "Creating vector of normal_distributions with " << max_thread_num << " normal_distributions.\n" << std::endl;
    }
    for (int thread_count = 0; thread_count < max_thread_num; ++thread_count)
    {
        std::normal_distribution<floatT> tmp_dist{0, 1};
        tmp_vec.emplace_back(tmp_dist);
    }
    return tmp_vec;
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
    HeatbathKernel               Heatbath1(Gluon1, GaugeAction::WilsonAction, distribution_uniform);
    OverrelaxationSubgroupKernel OverrelaxationSubgroup1(Gluon1, GaugeAction::WilsonAction);
    // auto end_copy = std::chrono::system_clock::now();
    // std::chrono::duration<double> copy_time = end_copy - start_copy;
    // std::cout << "Time for copy: " << copy_time.count() << std::endl;
    // Get old value of collective variable
    // TODO: Since the calculation is expensive in our case, we should try to reduce the number of CV calculations
    //       Instead of recomputing the CV, only compute the new CV and remember the old CV from last step
    // double CV_old {CV_function(Gluon1, Gluon2, rho_stout, 15)};
    // Perform update sweeps
    // HeatbathSU3(Gluon1, n_sweep_heatbath, distribution_uniform);
    // OverrelaxationSubgroupOld(Gluon1, n_sweep_orelax);
    // Update_function(Gluon1, distribution_uniform, n_sweep_heatbath, n_sweep_orelax);
    Iterator::Checkerboard(Heatbath1, n_sweep_heatbath);
    Iterator::Checkerboard(OverrelaxationSubgroup1, n_sweep_orelax);
    // Get new value of collective variable
    double CV_new {CV_function(Gluon1, Gluon2, Gluon3, n_smear_meta, rho_stout)};
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

// TODO: Old version, can probably delete this
// double MetaCharge(const GaugeField& Gluon, GaugeField& Gluon_copy1, GaugeField& Gluon_copy2, const int n_smear, const double smear_param)
// {
//     /*StoutSmearing4D(Gluon, Gluon1);*/
//     Gluon_copy1 = Gluon;
//     StoutSmearingN(Gluon_copy1, Gluon_copy2, n_smear, smear_param);
//     // TODO: Probably need to rewrite StoutSmearingN so we don't have to manually keep track
//     // For even n_smear, we need to use Gluon1, for odd n_smear we need to use Gluon2!
//     // See description of StoutSmearingN()
//     if (n_smear % 2 == 0)
//     {
//         return TopChargeGluonicSymm(Gluon_copy1);
//     }
//     else
//     {
//         return TopChargeGluonicSymm(Gluon_copy2);
//     }
//     // Old version
//     // Gluon_copy = Gluon;
//     // WilsonFlowForward(Gluon_copy, epsilon, n_flow);
//     // return TopChargeGluonicSymm(Gluon_copy);
// }

//-----
// Calculates and writes observables to logfile

void Observables(const GaugeField& Gluon, GaugeField& Gluonchain, std::ofstream& logstream, const int n_count, const int n_smear, const bool print_newline = true)
{
    vector<double>               Action(n_smear + 1);
    vector<double>               ActionImproved(n_smear + 1);
    vector<double>               ActionUnnormalized(n_smear + 1);
    vector<double>               Plaquette(n_smear + 1);
    vector<double>               EPlaqutte(n_smear + 1);
    vector<double>               EClover(n_smear + 1);
    vector<double>               WLoop2(n_smear + 1);
    vector<double>               WLoop4(n_smear + 1);
    vector<double>               WLoop8(n_smear + 1);
    vector<double>               PLoopRe(n_smear + 1);
    vector<double>               PLoopIm(n_smear + 1);
    vector<std::complex<double>> PLoop(n_smear + 1);
    // vector<double> TopologicalCharge(n_smear + 1);
    vector<double>               TopologicalChargeSymm(n_smear + 1);
    vector<double>               TopologicalChargeUnimproved(n_smear + 1);
    // auto ActionStruct = CreateObservable<double>(WilsonAction::ActionNormalized, n_smear + 1 , "Action");
    // GaugeAction::Rectangular<1> WAct(beta, 1.0, 0.0);
    GaugeAction::Rectangular<2> SymanzikAction(beta, 1.0 + 8.0 * 1.0/12.0, -1.0/12.0);

    // TODO: We can construct the Action in the argument list, so we do not have to declare it in a previous line
    //       HOWEVER this requires the constructor to take the Action as a forwarding reference. This way, the rvalue reference extends the lifetime of
    //       the temporary Action functor to be the same as the Flow functor.
    // TODO: This might be really convenient for many other functors! I should definitely go through all of them and see if I should take more arguments as
    //       forwarding references in the constructor. Also, check if more references should be made const members
    //       Actually not sure if this is correct anymore. Possibly, the lifetime only gets extended for the duration of the constructor! 
    Integrators::GradientFlow::Euler Flow_Integrator(Gluonsmeared2);
    // GradientFlowKernel Flow(Gluon, Gluonsmeared1, Gluonsmeared2, Flow_Integrator, GaugeAction::Rectangular<1>(beta, 1.0, 0.0), rho_stout);
    GradientFlowKernel Flow(Gluon, Gluonsmeared1, Flow_Integrator, GaugeAction::WilsonAction, rho_stout);

    // CoolingKernel Cooling(Gluonsmeared1);
    // GradientFlowKernel Cooling(Gluonsmeared1, 0.12);

    // Unsmeared observables
    // auto start_action = std::chrono::system_clock::now();
    Action[0]                      = WilsonAction::ActionNormalized(Gluon);
    // auto end_action = std::chrono::system_clock::now();
    // std::chrono::duration<double> action_time = end_action - start_action;
    // std::cout << "Time for calculating action: " << action_time.count() << std::endl;
    ActionImproved[0]              = SymanzikAction.ActionNormalized(Gluon);
    Plaquette[0]                   = PlaquetteSum(Gluon);
    EPlaqutte[0]                   = EnergyDensity::Plaquette(Gluon);
    FieldStrengthTensor::Clover(Gluon, F_tensor);
    EClover[0]                     = EnergyDensity::Clover(F_tensor);

    // auto start_wilson = std::chrono::system_clock::now();
    WLoop2[0]                      = WilsonLoop<0, 2,  true>(Gluon, Gluonchain);
    // auto end_wilson = std::chrono::system_clock::now();
    // std::chrono::duration<double> wilson_time = end_wilson - start_wilson;
    // std::cout << "Time for calculating wilson 2: " << wilson_time.count() << std::endl;

    // start_wilson = std::chrono::system_clock::now();
    WLoop4[0]                      = WilsonLoop<2, 4, false>(Gluon, Gluonchain);
    // end_wilson = std::chrono::system_clock::now();
    // wilson_time = end_wilson - start_wilson;
    // std::cout << "Time for calculating wilson 4: " << wilson_time.count() << std::endl;

    // start_wilson = std::chrono::system_clock::now();
    WLoop8[0]                      = WilsonLoop<4, 8, false>(Gluon, Gluonchain);
    // end_wilson = std::chrono::system_clock::now();
    // wilson_time = end_wilson - start_wilson;
    // std::cout << "Time for calculating wilson 8: " << wilson_time.count() << std::endl;

    // auto start_polyakov = std::chrono::system_clock::now();
    PLoop[0]                       = PolyakovLoop(Gluon);
    // auto end_polyakov = std::chrono::system_clock::now();
    // std::chrono::duration<double> polyakov_time = end_polyakov - start_polyakov;
    // std::cout << "Time for calculating Polyakov: " << polyakov_time.count() << std::endl;

    // auto start_topcharge = std::chrono::system_clock::now();
    // TopologicalCharge[0] = TopChargeGluonic(Gluon);
    // auto end_topcharge = std::chrono::system_clock::now();
    // std::chrono::duration<double> topcharge_time = end_topcharge - start_topcharge;
    // std::cout << "Time for calculating topcharge: " << topcharge_time.count() << std::endl;
    // auto start_topcharge_symm = std::chrono::system_clock::now();
    TopologicalChargeSymm[0]       = TopChargeGluonicSymm(Gluon);
    // TopologicalChargeSymm[0]       = CloverChargeFromF(F_tensor);
    // auto end_topcharge_symm = std::chrono::system_clock::now();
    // std::chrono::duration<double> topcharge_symm_time = end_topcharge_symm - start_topcharge_symm;
    // std::cout << "Time for calculating topcharge (symm): " << topcharge_symm_time.count() << std::endl;
    // auto start_topcharge_plaq = std::chrono::system_clock::now();
    TopologicalChargeUnimproved[0] = TopChargeGluonicUnimproved(Gluon);
    // auto end_topcharge_plaq = std::chrono::system_clock::now();
    // std::chrono::duration<double> topcharge_plaq_time = end_topcharge_plaq - start_topcharge_plaq;
    // std::cout << "Time for calculating topcharge (plaq): " << topcharge_plaq_time.count() << std::endl;
    // ActionStruct.Calculate(0, std::cref(Gluon));

    //-----
    // Begin smearing
    if (n_smear > 0)
    {
        // Apply smearing
        // auto start_smearing = std::chrono::system_clock::now();
        Flow(n_smear_skip);
        // Gluonsmeared1 = Gluon;
        // Iterator::Checkerboard(Cooling, 1);
        // WilsonFlowForward(Gluonsmeared1, 0.12, 1);
        // auto end_smearing = std::chrono::system_clock::now();
        // std::chrono::duration<double> smearing_time = end_smearing - start_smearing;
        // std::cout << "Time for calculating smearing: " << smearing_time.count() << std::endl;
        // Calculate observables
        Action[1]                      = WilsonAction::ActionNormalized(Gluonsmeared1);
        ActionImproved[1]              = SymanzikAction.ActionNormalized(Gluonsmeared1);
        Plaquette[1]                   = PlaquetteSum(Gluonsmeared1);
        EPlaqutte[1]                   = EnergyDensity::Plaquette(Gluonsmeared1);
        FieldStrengthTensor::Clover(Gluonsmeared1, F_tensor);
        EClover[1]                     = EnergyDensity::Clover(F_tensor);
        WLoop2[1]                      = WilsonLoop<0, 2,  true>(Gluonsmeared1, Gluonchain);
        WLoop4[1]                      = WilsonLoop<2, 4, false>(Gluonsmeared1, Gluonchain);
        WLoop8[1]                      = WilsonLoop<4, 8, false>(Gluonsmeared1, Gluonchain);
        PLoop[1]                       = PolyakovLoop(Gluonsmeared1);
        // TopologicalCharge[1] = TopChargeGluonic(Gluonsmeared1);
        TopologicalChargeSymm[1]       = TopChargeGluonicSymm(Gluonsmeared1);
        // TopologicalChargeSymm[1]       = CloverChargeFromF(F_tensor);
        TopologicalChargeUnimproved[1] = TopChargeGluonicUnimproved(Gluonsmeared1);
        // ActionStruct.Calculate(1, std::cref(Gluonsmeared1));
    }

    //-----
    // Further smearing steps
    for (int smear_count = 2; smear_count <= n_smear; ++smear_count)
    {
        // Even
        if (smear_count % 2 == 0)
        {
            // Apply smearing
            Flow.Resume(n_smear_skip);
            // Iterator::Checkerboard(Cooling, n_smear_skip);
            // WilsonFlowForward(Gluonsmeared1, 0.12, n_smear_skip);
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
            Action[smear_count]                      = WilsonAction::ActionNormalized(Gluonsmeared1);
            ActionImproved[smear_count]              = SymanzikAction.ActionNormalized(Gluonsmeared1);
            Plaquette[smear_count]                   = PlaquetteSum(Gluonsmeared1);
            EPlaqutte[smear_count]                   = EnergyDensity::Plaquette(Gluonsmeared1);
            FieldStrengthTensor::Clover(Gluonsmeared1, F_tensor);
            EClover[smear_count]                     = EnergyDensity::Clover(F_tensor);
            WLoop2[smear_count]                      = WilsonLoop<0, 2,  true>(Gluonsmeared1, Gluonchain);
            WLoop4[smear_count]                      = WilsonLoop<2, 4, false>(Gluonsmeared1, Gluonchain);
            WLoop8[smear_count]                      = WilsonLoop<4, 8, false>(Gluonsmeared1, Gluonchain);
            PLoop[smear_count]                       = PolyakovLoop(Gluonsmeared1);
            // TopologicalCharge[smear_count] = TopChargeGluonic(Gluonsmeared2);
            TopologicalChargeSymm[smear_count]       = TopChargeGluonicSymm(Gluonsmeared1);
            // TopologicalChargeSymm[smear_count]       = CloverChargeFromF(F_tensor);
            TopologicalChargeUnimproved[smear_count] = TopChargeGluonicUnimproved(Gluonsmeared1);
            // ActionStruct.Calculate(smear_count, std::cref(Gluonsmeared2));
        }
        // Odd
        else
        {
            // Apply smearing
            Flow.Resume(n_smear_skip);
            // Iterator::Checkerboard(Cooling, n_smear_skip);
            // WilsonFlowForward(Gluonsmeared1, 0.12, n_smear_skip);
            // Calculate observables
            Action[smear_count]                      = WilsonAction::ActionNormalized(Gluonsmeared1);
            ActionImproved[smear_count]              = SymanzikAction.ActionNormalized(Gluonsmeared1);
            Plaquette[smear_count]                   = PlaquetteSum(Gluonsmeared1);
            EPlaqutte[smear_count]                   = EnergyDensity::Plaquette(Gluonsmeared1);
            FieldStrengthTensor::Clover(Gluonsmeared1, F_tensor);
            EClover[smear_count]                     = EnergyDensity::Clover(F_tensor);
            WLoop2[smear_count]                      = WilsonLoop<0, 2,  true>(Gluonsmeared1, Gluonchain);
            WLoop4[smear_count]                      = WilsonLoop<2, 4, false>(Gluonsmeared1, Gluonchain);
            WLoop8[smear_count]                      = WilsonLoop<4, 8, false>(Gluonsmeared1, Gluonchain);
            PLoop[smear_count]                       = PolyakovLoop(Gluonsmeared1);
            // TopologicalCharge[smear_count] = TopChargeGluonic(Gluonsmeared1);
            TopologicalChargeSymm[smear_count]       = TopChargeGluonicSymm(Gluonsmeared1);
            // TopologicalChargeSymm[smear_count]       = CloverChargeFromF(F_tensor);
            TopologicalChargeUnimproved[smear_count] = TopChargeGluonicUnimproved(Gluonsmeared1);
            // ActionStruct.Calculate(smear_count, std::cref(Gluonsmeared1));
        }
    }

    //-----
    std::transform(Plaquette.begin(), Plaquette.end(), Plaquette.begin(), [&Gluon](const auto& element){return element / Gluon.Volume();});
    std::transform(Action.begin(), Action.end(), ActionUnnormalized.begin(), [&Gluon](const auto& element){return 6.0 * beta * Gluon.Volume() * element;});
    std::transform(PLoop.begin(), PLoop.end(), PLoopRe.begin(), [](const auto& element){return std::real(element);});
    std::transform(PLoop.begin(), PLoop.end(), PLoopIm.begin(), [](const auto& element){return std::imag(element);});

    //-----
    // Write to logfile
    std::time_t log_time {std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
    logstream << "[Step " << n_count << "] -" << std::put_time(std::localtime(&log_time), "%c") << "-\n";
    logstream << "Smoothing method: " << Flow.ReturnIntegratorName() << "\n";
    //-----
    if constexpr(n_hmc != 0)
    {
        logstream << "DeltaH: " << DeltaH << "\n";
    }
    if constexpr(metadynamics_enabled and tempering_enabled)
    {
        logstream << "DeltaVTempering: " << DeltaVTempering << "\n";
    }
    if constexpr(n_instanton_update != 0)
    {
        logstream << "DeltaSInstanton: " << DeltaSInstanton << "\n";
    }
    //-----
    logstream << "Wilson_Action: ";
    // std::copy(Action.cbegin(), std::prev(Action.cend()), std::ostream_iterator<double>(logstream, " "));
    std::copy(std::cbegin(Action), std::prev(std::cend(Action)), std::ostream_iterator<double>(logstream, " "));
    logstream << Action.back() << "\n";
    //-----
    logstream << "Improved_Action: ";
    // std::copy(ActionImproved.cbegin(), std::prev(ActionImproved.cend()), std::ostream_iterator<double>(logstream, " "));
    std::copy(std::cbegin(ActionImproved), std::prev(std::cend(ActionImproved)), std::ostream_iterator<double>(logstream, " "));
    logstream << ActionImproved.back() << "\n";
    //-----
    logstream << "Plaquette: ";
    // std::copy(Plaquette.cbegin(), std::prev(Plaquette.cend()), std::ostream_iterator<double>(logstream, " "));
    std::copy(std::cbegin(Plaquette), std::prev(std::cend(Plaquette)), std::ostream_iterator<double>(logstream, " "));
    logstream << Plaquette.back() << "\n";
    //-----
    logstream << "E_plaq: ";
    // std::copy(EPlaqutte.cbegin(), std::prev(EPlaqutte.cend()), std::ostream_iterator<double>(logstream, " "));
    std::copy(std::cbegin(EPlaqutte), std::prev(std::cend(EPlaqutte)), std::ostream_iterator<double>(logstream, " "));
    logstream << EPlaqutte.back() << "\n";
    //-----
    logstream << "E_clov: ";
    // std::copy(EClover.cbegin(), std::prev(EClover.cend()), std::ostream_iterator<double>(logstream, " "));
    std::copy(std::cbegin(EClover), std::prev(std::cend(EClover)), std::ostream_iterator<double>(logstream, " "));
    logstream << EClover.back() << "\n";
    //-----
    logstream << "Wilson_Action(unnormalized): ";
    // std::copy(Action.cbegin(), std::prev(Action.cend()), std::ostream_iterator<double>(logstream, " "));
    std::copy(std::cbegin(ActionUnnormalized), std::prev(std::cend(ActionUnnormalized)), std::ostream_iterator<double>(logstream, " "));
    logstream << ActionUnnormalized.back() << "\n";
    //-----
    logstream << "Wilson_loop(L=2): ";
    // std::copy(WLoop2.cbegin(), std::prev(WLoop2.cend()), std::ostream_iterator<double>(logstream, " "));
    std::copy(std::cbegin(WLoop2), std::prev(std::cend(WLoop2)), std::ostream_iterator<double>(logstream, " "));
    logstream << WLoop2.back() << "\n";
    //-----
    logstream << "Wilson_loop(L=4): ";
    // std::copy(WLoop4.cbegin(), std::prev(WLoop4.cend()), std::ostream_iterator<double>(logstream, " "));
    std::copy(std::cbegin(WLoop4), std::prev(std::cend(WLoop4)), std::ostream_iterator<double>(logstream, " "));
    logstream << WLoop4.back() << "\n";
    //-----
    logstream << "Wilson_loop(L=8): ";
    // std::copy(WLoop8.cbegin(), std::prev(WLoop8.cend()), std::ostream_iterator<double>(logstream, " "));
    std::copy(std::cbegin(WLoop8), std::prev(std::cend(WLoop8)), std::ostream_iterator<double>(logstream, " "));
    logstream << WLoop8.back() << "\n";
    //-----
    logstream << "Polyakov_loop(Re): ";
    std::copy(std::cbegin(PLoopRe), std::prev(std::cend(PLoopRe)), std::ostream_iterator<double>(logstream, " "));
    logstream << PLoopRe.back() << "\n";
    //-----
    logstream << "Polyakov_loop(Im): ";
    std::copy(std::cbegin(PLoopIm), std::prev(std::cend(PLoopIm)), std::ostream_iterator<double>(logstream, " "));
    logstream << PLoopIm.back() << "\n";
    //-----
    logstream << "TopChargeClov: ";
    std::copy(std::cbegin(TopologicalChargeSymm), std::prev(std::cend(TopologicalChargeSymm)), std::ostream_iterator<double>(logstream, " "));
    logstream << TopologicalChargeSymm.back() << "\n";
    //-----
    logstream << "TopChargePlaq: ";
    std::copy(std::cbegin(TopologicalChargeUnimproved), std::prev(std::cend(TopologicalChargeUnimproved)), std::ostream_iterator<double>(logstream, " "));
    logstream << TopologicalChargeUnimproved.back() << "\n";
    //-----
    if (print_newline)
    {
        logstream << std::endl;
    }
}

void Observables(const GaugeField& Gluon, GaugeField& Gluonchain, std::ofstream& logstream, const MetaBiasPotential& Metapotential, const int n_count, const int n_smear)
{
    // Call the regular Observables() function, but do not print a newline at the end, since we still want to log the current CV
    Observables(Gluon, Gluonchain, logstream, n_count, n_smear, false);

    double CV_current {Metapotential.ReturnCV_current()};
    logstream << "CV_MetaD: " << CV_current << "\n";
    logstream << "Metapotential: " << Metapotential.ReturnPotential(CV_current) << "\n" << std::endl;
}

//-----

int main()
{
    // iostream not synchronized with corresponding C streams, might cause a problem with C libraries and might not be thread safe
    std::ios_base::sync_with_stdio(false);
    std::cout << std::setprecision(12) << std::fixed;
    datalog << std::setprecision(12) << std::fixed;

    Configuration();
    prng_vector = CreatePRNGs();
    if constexpr(n_hmc != 0)
    {
        ndist_vector = CreateNormalDistributions();
    }

    // Default width of random numbers used in Metropolis update is 0.5
    floatT epsilon {0.5};

    std::uniform_real_distribution<floatT> distribution_prob(0.0, 1.0);
    std::uniform_real_distribution<floatT> distribution_uniform(0.0, 1.0);
    std::uniform_int_distribution<int> distribution_choice(1, 8);
    std::uniform_int_distribution<int> distribution_instanton(0, 1);

    CreateFiles();
    Gluon.SetToIdentity();

    auto startcalc {std::chrono::system_clock::now()};
    datalog.open(logfilepath, std::fstream::out | std::fstream::app);
    wilsonlog.open(wilsonfilepath, std::fstream::out | std::fstream::app);

    // Commonly used gauge actions
    GaugeAction::WilsonAction.SetBeta(beta);
    GaugeAction::LüscherWeiszAction.SetBeta(beta);
    GaugeAction::IwasakiAction.SetBeta(beta);
    GaugeAction::DBW2Action.SetBeta(beta);

    // Initialize update functors
    HeatbathKernel                   Heatbath(Gluon, GaugeAction::DBW2Action, distribution_uniform);
    // OverrelaxationDirectKernel       OverrelaxationDirect(Gluon, distribution_prob);
    OverrelaxationSubgroupKernel     OverrelaxationSubgroup(Gluon, GaugeAction::DBW2Action);
    Integrators::HMC::OMF_4          OMF_4_Integrator;
    // Integrators::HMC::Leapfrog_OMF_4 LFRG_OMF_4_Integrator;
    // Integrators::HMC::OMF_2_OMF_4    OMF_2_OMF_4_Integrator;
    GaugeUpdates::HMCKernel          HMC(Gluon, Gluonsmeared1, Gluonsmeared2, OMF_4_Integrator, GaugeAction::DBW2Action, distribution_prob);

    // LoadConfigBMW(Gluon, "GradientFlowBMW/conf0001.conf");

    // Regular updates without Metadynamics
    if constexpr(!metadynamics_enabled)
    {
        // When using HMC, the thermalization is done without accept-reject step
        if constexpr(n_hmc != 0)
        {
            datalog << "[HMC start thermalization]\n";
            for (int n_count = 0; n_count < 20; ++n_count)
            {
                HMC(10, false);
            }
            datalog << "[HMC end thermalization]\n" << std::endl;
        }
        else
        {
            for (int n_count = 0; n_count < 20; ++n_count)
            {
                Iterator::Checkerboard4(Heatbath, n_heatbath);
                Iterator::Checkerboard4(OverrelaxationSubgroup, n_orelax);
            }
        }

        for (int n_count = 0; n_count < n_run; ++n_count)
        {
            // auto start_update_metro {std::chrono::system_clock::now()};
            if constexpr(n_metro != 0 and multi_hit != 0)
            {
                std::uniform_real_distribution<floatT> distribution_unitary(-epsilon, epsilon);
                MetropolisKernel Metropolis(Gluon, GaugeAction::DBW2Action, multi_hit, distribution_prob, distribution_unitary, distribution_choice);
                Iterator::Checkerboard4Sum(Metropolis, acceptance_count, n_metro);
                // TODO: Perhaps this should all happen automatically inside the functor?
                //       At the very least, we should probably combine the two actions below into one function
                epsilon = Metropolis.AdjustedEpsilon(epsilon, acceptance_count);
                acceptance_count = 0;
                // MetropolisUpdate(Gluon, n_metro, acceptance_count, epsilon, distribution_prob, distribution_choice, distribution_unitary);
            }
            // auto end_update_metro {std::chrono::system_clock::now()};
            // std::chrono::duration<double> update_time_metro {end_update_metro - start_update_metro};
            // std::cout << "Time for " << n_metro << " Metropolis updates: " << update_time_metro.count() << std::endl;
            //-----
            // auto start_update_heatbath {std::chrono::system_clock::now()};
            if constexpr(n_heatbath != 0)
            {
                // HeatbathSU3(Gluon, n_heatbath, distribution_uniform);
                Iterator::Checkerboard4(Heatbath, n_heatbath);
            }
            // auto end_update_heatbath {std::chrono::system_clock::now()};
            // std::chrono::duration<double> update_time_heatbath {end_update_heatbath - start_update_heatbath};
            // std::cout << "Time for " << n_heatbath << " heatbath updates: " << update_time_heatbath.count() << std::endl;
            //-----
            // auto start_update_hmc {std::chrono::system_clock::now()};
            if constexpr(n_hmc != 0)
            {
                HMC(n_hmc, true);
            }
            // auto end_update_hmc {std::chrono::system_clock::now()};
            // std::chrono::duration<double> update_time_hmc {end_update_hmc - start_update_hmc};
            // std::cout << "Time for one HMC trajectory: " << update_time_hmc.count() << std::endl;
            //-----
            // auto start_update_or = std::chrono::system_clock::now();
            if constexpr(n_orelax != 0)
            {
                // double action_before {GaugeAction::WilsonAction.Action(Gluon)};
                // Iterator::CheckerboardSum(OverrelaxationDirect, acceptance_count_or, n_orelax);
                Iterator::Checkerboard4(OverrelaxationSubgroup, n_orelax);
                // double action_after {GaugeAction::WilsonAction.Action(Gluon)};
                // std::cout << "Action (before): " << action_before << std::endl;
                // std::cout << "Action (after): " << action_after << std::endl;
                // std::cout << action_after - action_before << std::endl;
            }
            // auto end_update_or = std::chrono::system_clock::now();
            // std::chrono::duration<double> update_time_or {end_update_or - start_update_or};
            // std::cout << "Time for " << n_orelax << " OR updates: " << update_time_or.count() << std::endl;
            //-----
            if constexpr(n_instanton_update != 0)
            {
                int        Q_instanton {distribution_instanton(prng_vector[omp_get_thread_num()]) * 2 - 1};
                int        L_half      {Nt/2 - 1};
                site_coord center      {L_half, L_half, L_half, L_half};
                int        radius      {5};
                // If the function is called for the first time, create Q = +1 and Q = -1 instanton configurations, otherwise reuse old configurations
                if (n_count == 0)
                {
                    BPSTInstantonUpdate(Gluon, Gluonsmeared1, Q_instanton, center, radius, acceptance_count_instanton, true, distribution_prob, true);
                }
                else
                {
                    BPSTInstantonUpdate(Gluon, Gluonsmeared1, Q_instanton, center, radius, acceptance_count_instanton, true, distribution_prob, false);
                }
            }
            //-----
            if (n_count % expectation_period == 0)
            {
                // auto start_observable = std::chrono::system_clock::now();
                Observables(Gluon, Gluonchain, datalog, n_count, n_smear);
                // auto end_observable = std::chrono::system_clock::now();
                // std::chrono::duration<double> observable_time {end_observable - start_observable};
                // std::cout << "Time for calculating observables: " << observable_time.count() << std::endl;

                // n_smear = 300;
                // n_smear_skip = 1;
                // Observables(Gluon, Gluonchain, datalog, n_count, n_smear, 0.12);

                // n_smear = 300;
                // n_smear_skip = 1;
                // Observables(Gluon, Gluonchain, datalog, n_count, n_smear, 0.08);

                // n_smear = 300;
                // n_smear_skip = 2;
                // Observables(Gluon, Gluonchain, datalog, n_count, n_smear, 0.04);

                // n_smear = 300;
                // n_smear_skip = 4;
                // Observables(Gluon, Gluonchain, datalog, n_count, n_smear, 0.02);

                // n_smear = 300;
                // n_smear_skip = 8;
                // Observables(Gluon, Gluonchain, datalog, n_count, n_smear, 0.01);

                // n_smear = 300;
                // n_smear_skip = 16;
                // Observables(Gluon, Gluonchain, datalog, n_count, n_smear, 0.005);
            }
        }
    }

    // Updates with Metadynamics
    if constexpr(metadynamics_enabled and !tempering_enabled)
    {
        // CV_min, CV_max, bin_number, weight, threshold_weight
        MetaBiasPotential TopBiasPotential{-8, 8, 800, 0.05, 1000.0};
        TopBiasPotential.LoadPotential("SU(3)_N=20x20x20x20_beta=1.250000/metapotential.txt");
        // TopBiasPotential.LoadPotential("metapotential.txt");
        // TopBiasPotential.SymmetrizePotential();
        TopBiasPotential.SymmetrizePotentialMaximum();
        // TopBiasPotential.Setweight(0.005);
        TopBiasPotential.SaveMetaParameters(metapotentialfilepath);
        TopBiasPotential.SaveMetaPotential(metapotentialfilepath);

        // GaugeUpdates::HMCMetaDKernel HMC_MetaD(Gluon, Gluonsmeared1, Gluonsmeared2, TopBiasPotential, OMF_2_OMF_4_Integrator, GaugeAction::DBW2Action, n_smear_meta, distribution_prob);
        GaugeUpdates::HMCMetaDData   MetadynamicsData(n_smear_meta);
        GaugeUpdates::HMCMetaDKernel HMC_MetaD(Gluon, Gluonsmeared1, Gluonsmeared2, TopBiasPotential, MetadynamicsData, OMF_4_Integrator, GaugeAction::DBW2Action, distribution_prob);

        // Thermalize with normal HMC
        datalog << "[HMC start thermalization]\n";
        for (int i = 0; i < 20; ++i)
        {
            // Iterator::Checkerboard(Heatbath, 1);
            // Iterator::Checkerboard(OverrelaxationSubgroup, 4);
            HMC(10, false);
        }
        datalog << "[HMC end thermalization]\n" << std::endl;

        for (int n_count = 0; n_count < n_run; ++n_count)
        {
            // auto start_update_meta = std::chrono::system_clock::now();
            HMC_MetaD(n_hmc, true);
            // MetadynamicsLocal(Gluon, Gluonsmeared1, Gluonsmeared2, Gluonsmeared3, TopBiasPotential, MetaCharge, CV, n_heatbath, n_orelax, distribution_prob, distribution_uniform);
            // auto end_update_meta = std::chrono::system_clock::now();
            // std::chrono::duration<double> update_time_meta {end_update_meta - start_update_meta};
            // std::cout << "Time for meta update: " << update_time_meta.count() << std::endl;
            if (n_count % expectation_period == 0)
            {
                Observables(Gluon, Gluonchain, datalog, TopBiasPotential, n_count, n_smear);
                if constexpr(metapotential_updated)
                {
                    if (n_count % (1 * expectation_period) == 0)
                    TopBiasPotential.SaveMetaPotential(metapotentialfilepath);
                }
            }
        }
    }

    // Updates with Metadynamics and parallel tempering
    if constexpr(metadynamics_enabled and tempering_enabled)
    {
        // For now limit ourselves to tempering between two streams, so we only need one additional gaugefield Gluon_temper:
        //     Gluon:        Regular local updates
        //     Gluon_temper: MetaD-HMC updates
        GaugeField                   Gluon_temper;
        // Need to set all links to identity, otherwise the links are zero matrices, which leads to NaNs during the HMC
        Gluon_temper.SetToIdentity();

        // Setup second ofstream for Gluon_temper (Gluon uses the default stream datalog)
        std::ofstream datalog_temper;
        datalog_temper << std::setprecision(12) << std::fixed;
        std::string logfilepath_temper = directoryname + "/log_temper.txt";
        datalog_temper.open(logfilepath_temper, std::fstream::out | std::fstream::app);

        // Conventional HMC only used during thermalization of Gluon_temper
        GaugeUpdates::HMCKernel                   HMC_temper(Gluon_temper, Gluonsmeared1, Gluonsmeared2, OMF_4_Integrator, GaugeAction::DBW2Action, distribution_prob);

        // CV_min, CV_max, bin_number, weight, threshold_weight
        MetaBiasPotential                         TopBiasPotential{-8, 8, 800, 0.05, 1000.0};
        TopBiasPotential.LoadPotential("metapotential_16_1.24.txt");
        TopBiasPotential.SymmetrizePotentialMaximum();
        TopBiasPotential.SaveMetaParameters(metapotentialfilepath);
        TopBiasPotential.SaveMetaPotential(metapotentialfilepath);
        GaugeUpdates::HMCMetaDData                MetadynamicsData(n_smear_meta);
        GaugeUpdates::HMCMetaDKernel              HMC_MetaD(Gluon_temper, Gluonsmeared1, Gluonsmeared2, TopBiasPotential, MetadynamicsData, OMF_4_Integrator, GaugeAction::DBW2Action, distribution_prob);

        GaugeUpdates::MetadynamicsTemperingKernel ParallelTemperingSwap(Gluon, Gluon_temper, Gluonsmeared1, Gluonsmeared2, TopBiasPotential, distribution_prob);

        // Thermalize Gluon with local updates, and Gluon_temper with normal HMC
        datalog << "[HMC start thermalization]\n";
        for (int i = 0; i < 20; ++i)
        {
            Iterator::Checkerboard4(Heatbath, n_heatbath);
            Iterator::Checkerboard4(OverrelaxationSubgroup, n_orelax);
            HMC_temper(10, false);
        }
        datalog << "[HMC end thermalization]\n" << std::endl;

        for (int n_count = 0; n_count < n_run; ++n_count)
        {
            // Perform updates on Gluon (for every MetaD-HMC update perform tempering_nonmetadynamics_sweeps updates on the config without Metadynamics)
            for (int n_count_nometa = 0; n_count_nometa < tempering_nonmetadynamics_sweeps; ++n_count_nometa)
            {
                Iterator::Checkerboard4(Heatbath, n_heatbath);
                Iterator::Checkerboard4(OverrelaxationSubgroup, n_orelax);
            }

            // Perform updates on Gluon_temper
            HMC_MetaD(n_hmc, true);

            // Propose tempering swap
            if (n_count % tempering_swap_period == 0)
            {
                datalog_temper << "Tempering swap accepted: " << ParallelTemperingSwap() << std::endl;
            }

            if (n_count % expectation_period == 0)
            {
                Observables(Gluon, Gluonchain, datalog, n_count, n_smear);
                Observables(Gluon_temper, Gluonchain, datalog_temper, TopBiasPotential, n_count, n_smear);
                if constexpr(metapotential_updated)
                {
                    if (n_count % (1 * expectation_period) == 0)
                    TopBiasPotential.SaveMetaPotential(metapotentialfilepath);
                }
            }
        }
        datalog_temper.close();
        datalog_temper.clear();
    }

    auto end {std::chrono::system_clock::now()};
    std::chrono::duration<double> elapsed_seconds {end - startcalc};
    std::time_t end_time {std::chrono::system_clock::to_time_t(end)};

    //-----
    // Print acceptance rates, PRNG width, and required time to terminal and to files

    std::cout << "\n";
    PrintFinal(std::cout, acceptance_count, acceptance_count_or, acceptance_count_hmc, acceptance_count_metadynamics_hmc, acceptance_count_tempering, epsilon, end_time, elapsed_seconds);

    PrintFinal(datalog, acceptance_count, acceptance_count_or, acceptance_count_hmc, acceptance_count_metadynamics_hmc, acceptance_count_tempering, epsilon, end_time, elapsed_seconds);
    datalog.close();
    datalog.clear();

    datalog.open(parameterfilepath, std::fstream::out | std::fstream::app);
    PrintFinal(datalog, acceptance_count, acceptance_count_or, acceptance_count_hmc, acceptance_count_metadynamics_hmc, acceptance_count_tempering, epsilon, end_time, elapsed_seconds);
    datalog.close();
    datalog.clear();

    PrintFinal(wilsonlog, acceptance_count, acceptance_count_or, acceptance_count_hmc, acceptance_count_metadynamics_hmc, acceptance_count_tempering, epsilon, end_time, elapsed_seconds);
    wilsonlog.close();
    wilsonlog.clear();
}
