#ifndef LETTUCE_PARAMETER_IO_HPP
#define LETTUCE_PARAMETER_IO_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "ansi_colors.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <string>
#include <typeinfo>
//----------------------------------------
// Standard C headers
#include <ctime>

//-----
// Check if specified command line argument exists

bool CheckCommandLineArgument(char** begin, char** end, const std::string& argument)
{
    return std::find(begin, end, argument) != end;
}

//-----
// Get parameter from specified command line argument (if it exists)

std::string ExtractCommandLineArgument(char** begin, char** end, const std::string& argument)
{
    char** iterator = std::find(begin, end, argument);
    if (iterator != end and ++iterator != end)
    {
        return std::string(*iterator);
    }
    else
    {
        return std::string("");
    }
}

//-----
// Function to get user input with error handling
// TODO: Constrain target to writeable range or something like that
// TODO: Should probably make clear that this version works for the terminal only
//       Rename this to ValidatedInTerminal and write alternative version for reading from files?

template<typename T>
void ValidatedIn(const std::string& message, T& target)
{
    // Keep count of tries and abort after too many tries (e.g. important when using nohup)
    int count {0};
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
// TODO: In the future, replace this with a function (or two functions) that can read parameters either from the terminal or from a file

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
    std::cout << "hmc_trajectory_length is "       << hmc_trajectory_length << ".\n";
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
    std::ofstream stream(filename, std::fstream::out | std::fstream::app);
    stream << std::setprecision(12) << std::fixed;
    stream << program_version << "\n";
    stream << "logfile\n\n";
    #ifdef FIXED_SEED
    stream << "FIXED_SEED\n";
    #endif
    stream << starttimestring << "\n";
    stream << "START_PARAMS\n";
    stream << "Gauge field precision = "            << typeid(floatT).name()            << "\n";
    stream << "Nt = "                               << Nt                               << "\n";
    stream << "Nx = "                               << Nx                               << "\n";
    stream << "Ny = "                               << Ny                               << "\n";
    stream << "Nz = "                               << Nz                               << "\n";
    stream << "beta = "                             << beta                             << "\n";
    stream << "n_run = "                            << n_run                            << "\n";
    stream << "expectation_period = "               << expectation_period               << "\n";
    stream << "n_smear = "                          << n_smear                          << "\n";
    stream << "n_smear_skip = "                     << n_smear_skip                     << "\n";
    stream << "rho_stout = "                        << rho_stout                        << "\n";
    stream << "n_metro = "                          << n_metro                          << "\n";
    stream << "multi_hit = "                        << multi_hit                        << "\n";
    stream << "metro_target_acceptance = "          << metro_target_acceptance          << "\n";
    stream << "n_heatbath = "                       << n_heatbath                       << "\n";
    stream << "n_hmc = "                            << n_hmc                            << "\n";
    stream << "hmc_trajectory_length = "            << hmc_trajectory_length            << "\n";
    stream << "n_orelax = "                         << n_orelax                         << "\n";
    stream << "n_instanton_update = "               << n_instanton_update               << "\n";
    stream << "metadynamics_enabled = "             << metadynamics_enabled             << "\n";
    stream << "metapotential_updated = "            << metapotential_updated            << "\n";
    stream << "n_smear_meta = "                     << n_smear_meta                     << "\n";
    stream << "tempering_enabled = "                << tempering_enabled                << "\n";
    stream << "tempering_nonmetadynamics_sweeps = " << tempering_nonmetadynamics_sweeps << "\n";
    stream << "tempering_swap_period = "            << tempering_swap_period            << "\n";
    stream << "END_PARAMS\n"                        << std::endl;
    stream.close();
    stream.clear();
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
    checkpointdirectory   = directoryname + "/checkpoints"; 
    std::filesystem::create_directory(directoryname);
    std::filesystem::create_directory(checkpointdirectory);
    std::cout << "\n\n" << "Created directory \"" << directoryname << "\".\n";
    logfilepath           = directoryname + "/log.txt";
    parameterfilepath     = directoryname + "/parameters.txt";
    wilsonfilepath        = directoryname + "/wilson.txt";
    metapotentialfilepath = directoryname + "/metapotential.txt";
    std::cout << Lettuce::Color::BoldBlue << "Filepath (log):\t\t"      << logfilepath                                    << "\n";
    std::cout                             << "Filepath (parameters):\t" << parameterfilepath                              << "\n";
    std::cout                             << "Filepath (wilson):\t"     << wilsonfilepath                                 << "\n";
    std::cout                             << "Filepath (metadyn):\t"    << metapotentialfilepath                          << "\n";
    std::cout                             << "Filepath (final):\t"      << checkpointdirectory   << Lettuce::Color::Reset << "\n";

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

#endif // LETTUCE_PARAMETER_IO_HPP
