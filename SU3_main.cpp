// Pure SU(3) theory
// Command line flags: -DFIXED_SEED

// #define EIGEN_USE_MKL_ALL

// Non-standard library headers
// Include these three header files first in this order
// TODO: Should probably check all includes and add the appropriate includes to all files
#include "LettuceGauge/defines.hpp"
#include "LettuceGauge/coords.hpp"
#include "LettuceGauge/lattice.hpp"
//-----
// Remaining files in alphabetic order (for now)
#include "LettuceGauge/actions/gauge/rectangular_action.hpp"
#include "LettuceGauge/IO/ansi_colors.hpp"
#include "LettuceGauge/IO/config_io/bmw_format.hpp"
#include "LettuceGauge/IO/config_io/bridge_text_format.hpp"
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

//-----

// GaugeField     Gluon         {Nt, Nx, Ny, Nz};
// GaugeField     Gluonsmeared1 {Nt, Nx, Ny, Nz};
// GaugeField     Gluonsmeared2 {Nt, Nx, Ny, Nz};
// GaugeField     Gluonchain    {Nt, Nx, Ny, Nz};

GaugeField                   Gluon;
GaugeField                   Gluonsmeared1;
GaugeField                   Gluonsmeared2;
GaugeField                   Gluonchain;
FullTensor                   F_tensor;

//-------------------------------------------------------------------------------------
// Calculates and writes observables to logfile

void Observables(const GaugeField& Gluon, GaugeField& Gluonchain, std::ofstream& logstream, const int n_count, const int n_smear, const double smearing_parameter = rho_stout, const bool print_newline = true)
{
    std::vector<double>               ActionImproved(n_smear + 1);
    std::vector<double>               Plaquette(n_smear + 1);
    std::vector<std::vector<double>>  ECloverTimeslice(n_smear + 1, std::vector<double>(Gluon.Length(0), 0.0));
    std::vector<double>               EClover(n_smear + 1);
    std::vector<double>               WLoop2(n_smear + 1);
    std::vector<double>               WLoop4(n_smear + 1);
    std::vector<double>               WLoop8(n_smear + 1);
    std::vector<std::complex<double>> PLoop(n_smear + 1);
    std::vector<double>               PLoopRe(n_smear + 1);
    std::vector<double>               PLoopIm(n_smear + 1);
    // std::vector<double> TopologicalChargeCloverSlow(n_smear + 1);
    std::vector<std::vector<double>>  TopologicalChargeCloverTimeslice(n_smear + 1, std::vector<double>(Gluon.Length(0), 0.0));
    std::vector<double>               TopologicalChargeClover(n_smear + 1);
    std::vector<double>               TopologicalChargePlaquette(n_smear + 1);
    // Calculate later from plaquette
    // std::vector<std::vector<double>>  EPlaquetteTimeslice(n_smear + 1, std::vector<double>(Gluon.Length(0), 0.0));
    std::vector<double>               EPlaquette(n_smear + 1);
    std::vector<double>               Action(n_smear + 1);
    std::vector<double>               ActionUnnormalized(n_smear + 1);
    // auto ActionStruct = CreateObservable<double>(WilsonAction::ActionNormalized, n_smear + 1 , "Action");
    // GaugeAction::Rectangular<1> WAct(beta, 1.0, 0.0);
    GaugeAction::Rectangular<2> SymanzikAction(beta, 1.0 + 8.0 * 1.0/12.0, -1.0/12.0);

    Integrators::GradientFlow::Euler Flow_Integrator(Gluonsmeared2);
    GradientFlowKernel Flow(Gluon, Gluonsmeared1, Flow_Integrator, GaugeAction::WilsonAction, smearing_parameter);

    // CoolingKernel Cooling(Gluonsmeared1);
    // GradientFlowKernel Cooling(Gluonsmeared1, 0.12);

    // Unsmeared observables
    // auto start_action = std::chrono::high_resolution_clock::now();
    // Action[0]                      = WilsonAction::ActionNormalized(Gluon);
    // auto end_action = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> action_time = end_action - start_action;
    // std::cout << "Time for calculating action: " << action_time.count() << std::endl;
    ActionImproved[0]              = SymanzikAction.ActionNormalized(Gluon);
    Plaquette[0]                   = PlaquetteSum(Gluon);
    FieldStrengthTensor::CloverTraceless(Gluon, F_tensor);
    for (int t = 0; t < Gluon.Length(0); ++t)
    {
        ECloverTimeslice[0][t]     = EnergyDensity::CloverTimeslice(F_tensor, t);
        // EPlaquetteTimeslice[0][t]  = EnergyDensity::PlaquetteTimeslice(Gluon, t);
    }
    // EClover[0]                     = EnergyDensity::Clover(F_tensor);
    EClover[0]                     = std::accumulate(ECloverTimeslice[0].cbegin(), ECloverTimeslice[0].cend(), 0.0);

    // auto start_wilson = std::chrono::high_resolution_clock::now();
    WLoop2[0]                      = WilsonLoop<0, 2,  true>(Gluon, Gluonchain);
    // auto end_wilson = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> wilson_time = end_wilson - start_wilson;
    // std::cout << "Time for calculating wilson 2: " << wilson_time.count() << std::endl;

    // start_wilson = std::chrono::high_resolution_clock::now();
    WLoop4[0]                      = WilsonLoop<2, 4, false>(Gluon, Gluonchain);
    // end_wilson = std::chrono::high_resolution_clock::now();
    // wilson_time = end_wilson - start_wilson;
    // std::cout << "Time for calculating wilson 4: " << wilson_time.count() << std::endl;

    // start_wilson = std::chrono::high_resolution_clock::now();
    WLoop8[0]                      = WilsonLoop<4, 8, false>(Gluon, Gluonchain);
    // end_wilson = std::chrono::high_resolution_clock::now();
    // wilson_time = end_wilson - start_wilson;
    // std::cout << "Time for calculating wilson 8: " << wilson_time.count() << std::endl;

    // auto start_polyakov = std::chrono::high_resolution_clock::now();
    PLoop[0]                       = PolyakovLoop(Gluon);
    // auto end_polyakov = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> polyakov_time = end_polyakov - start_polyakov;
    // std::cout << "Time for calculating Polyakov: " << polyakov_time.count() << std::endl;

    // auto start_topcharge = std::chrono::high_resolution_clock::now();
    // TopologicalChargeCloverSlow[0] = TopChargeCloverSlow(Gluon);
    // auto end_topcharge = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> topcharge_time = end_topcharge - start_topcharge;
    // std::cout << "Time for calculating topcharge: " << topcharge_time.count() << std::endl;
    // auto start_topcharge_timeslice = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < Gluon.Length(0); ++t)
    {
        TopologicalChargeCloverTimeslice[0][t] = TopChargeCloverTimeslice(Gluon, t);
    }
    // auto end_topcharge_timeslice = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> topcharge_timeslice_time = end_topcharge_timeslice - start_topcharge_timeslice;
    // std::cout << "Time for calculating topcharge timeslice: " << topcharge_timeslice_time.count() << std::endl;
    // auto start_topcharge_symm = std::chrono::high_resolution_clock::now();
    TopologicalChargeClover[0]     = std::accumulate(TopologicalChargeCloverTimeslice[0].cbegin(), TopologicalChargeCloverTimeslice[0].cend(), 0.0);
    // TopologicalChargeClover[0]     = TopChargeClover(Gluon);
    // TopologicalChargeClover[0]     = TopologicalCharge::CloverChargeFromFTensor(F_tensor);
    // auto end_topcharge_symm = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> topcharge_symm_time = end_topcharge_symm - start_topcharge_symm;
    // std::cout << "Time for calculating topcharge (symm): " << topcharge_symm_time.count() << std::endl;
    // auto start_topcharge_plaq = std::chrono::high_resolution_clock::now();
    TopologicalChargePlaquette[0]  = TopChargePlaquette(Gluon);
    // auto end_topcharge_plaq = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> topcharge_plaq_time = end_topcharge_plaq - start_topcharge_plaq;
    // std::cout << "Time for calculating topcharge (plaq): " << topcharge_plaq_time.count() << std::endl;
    // ActionStruct.Calculate(0, std::cref(Gluon));

    //-----
    // Measurements involving smearing
    for (int smear_count = 1; smear_count <= n_smear; ++smear_count)
    {
        // Apply smearing (first call is distinct from the calls afterwards, since we need to copy the unsmeared gaugefield here, but not later on)
        if (smear_count == 1)
        {
            // auto start_smear = std::chrono::high_resolution_clock::now();
            Flow(n_smear_skip);
            // auto end_smear = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> smear_time = end_smear - start_smear;
            // std::cout << "Time for smearing with " << Flow.ReturnIntegratorName() << ": " << smear_time.count() << std::endl;
        }
        else
        {
            Flow.Resume(n_smear_skip);
        }
        // Iterator::Checkerboard(Cooling, n_smear_skip);
        // Calculate observables
        ActionImproved[smear_count]              = SymanzikAction.ActionNormalized(Gluonsmeared1);
        Plaquette[smear_count]                   = PlaquetteSum(Gluonsmeared1);
        FieldStrengthTensor::CloverTraceless(Gluonsmeared1, F_tensor);
        for (int t = 0; t < Gluon.Length(0); ++t)
        {
            ECloverTimeslice[smear_count][t]               = EnergyDensity::CloverTimeslice(F_tensor, t);
            // EPlaquetteTimeslice[smear_count][t]  = EnergyDensity::PlaquetteTimeslice(Gluonsmeared1, t);
        }
        // EClover[smear_count]                     = EnergyDensity::Clover(F_tensor);
        EClover[smear_count]                     = std::accumulate(ECloverTimeslice[smear_count].cbegin(), ECloverTimeslice[smear_count].cend(), 0.0);
        WLoop2[smear_count]                      = WilsonLoop<0, 2,  true>(Gluonsmeared1, Gluonchain);
        WLoop4[smear_count]                      = WilsonLoop<2, 4, false>(Gluonsmeared1, Gluonchain);
        WLoop8[smear_count]                      = WilsonLoop<4, 8, false>(Gluonsmeared1, Gluonchain);
        PLoop[smear_count]                       = PolyakovLoop(Gluonsmeared1);
        // TopologicalChargeCloverSlow[smear_count] = TopChargeCloverSlow(Gluonsmeared2);
        for (int t = 0; t < Gluon.Length(0); ++t)
        {
            TopologicalChargeCloverTimeslice[smear_count][t] = TopChargeCloverTimeslice(Gluonsmeared1, t);
        }
        TopologicalChargeClover[smear_count]     = std::accumulate(TopologicalChargeCloverTimeslice[smear_count].cbegin(), TopologicalChargeCloverTimeslice[smear_count].cend(), 0.0);
        // TopologicalChargeClover[smear_count]     = TopChargeClover(Gluonsmeared1);
        // TopologicalChargeClover[smear_count]     = TopologicalCharge::CloverChargeFromFTensor(F_tensor);
        TopologicalChargePlaquette[smear_count]  = TopChargePlaquette(Gluonsmeared1);
        // ActionStruct.Calculate(smear_count, std::cref(Gluonsmeared2));
    }

    //-----
    // Final processing of observables (extracting real and imaginary parts, normalizing, ...)
    // These computations are trivial and can always be done on the host, so simply use std::transform for convenience here
    std::transform(Plaquette.cbegin(), Plaquette.cend(),          Plaquette.begin(), [&Gluon](const auto& element){return element / Gluon.Volume();});
    std::transform(Plaquette.cbegin(), Plaquette.cend(),         EPlaquette.begin(), [      ](const auto& element){return 36.0 - 2.0 * element;});
    std::transform(Plaquette.cbegin(), Plaquette.cend(),             Action.begin(), [      ](const auto& element){return 1.0 - element / 18.0;});
    std::transform(   Action.cbegin(),    Action.cend(), ActionUnnormalized.begin(), [&Gluon](const auto& element){return 6.0 * beta * Gluon.Volume() * element;});
    std::transform(    PLoop.cbegin(),     PLoop.cend(),            PLoopRe.begin(), [      ](const auto& element){return std::real(element);});
    std::transform(    PLoop.cbegin(),     PLoop.cend(),            PLoopIm.begin(), [      ](const auto& element){return std::imag(element);});

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
    // std::copy(EPlaquette.cbegin(), std::prev(EPlaquette.cend()), std::ostream_iterator<double>(logstream, " "));
    std::copy(std::cbegin(EPlaquette), std::prev(std::cend(EPlaquette)), std::ostream_iterator<double>(logstream, " "));
    logstream << EPlaquette.back() << "\n";
    //-----
    // logstream << "E_plaq_timeslice: ";
    // for (int smear_count = 0; smear_count < n_smear; ++smear_count)
    // {
    //     std::copy(std::cbegin(EPlaquetteTimeslice[smear_count]), std::cend(EPlaquetteTimeslice[smear_count]), std::ostream_iterator<double>(logstream, " "));
    // }
    // std::copy(std::cbegin(EPlaquetteTimeslice[n_smear]), std::prev(std::cend(EPlaquetteTimeslice[n_smear])), std::ostream_iterator<double>(logstream, " "));
    // logstream << EPlaquetteTimeslice[n_smear].back() << "\n";
    //-----
    logstream << "E_clov: ";
    // std::copy(EClover.cbegin(), std::prev(EClover.cend()), std::ostream_iterator<double>(logstream, " "));
    std::copy(std::cbegin(EClover), std::prev(std::cend(EClover)), std::ostream_iterator<double>(logstream, " "));
    logstream << EClover.back() << "\n";
    //-----
    logstream << "E_clov_timeslice: ";
    for (int smear_count = 0; smear_count < n_smear; ++smear_count)
    {
        std::copy(std::cbegin(ECloverTimeslice[smear_count]), std::cend(ECloverTimeslice[smear_count]), std::ostream_iterator<double>(logstream, " "));
    }
    std::copy(std::cbegin(ECloverTimeslice[n_smear]), std::prev(std::cend(ECloverTimeslice[n_smear])), std::ostream_iterator<double>(logstream, " "));
    logstream << ECloverTimeslice[n_smear].back() << "\n";
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
    std::copy(std::cbegin(TopologicalChargeClover), std::prev(std::cend(TopologicalChargeClover)), std::ostream_iterator<double>(logstream, " "));
    logstream << TopologicalChargeClover.back() << "\n";
    //-----
    logstream << "TopChargeClov_timeslice: ";
    for (int smear_count = 0; smear_count < n_smear; ++smear_count)
    {
        std::copy(std::cbegin(TopologicalChargeCloverTimeslice[smear_count]), std::cend(TopologicalChargeCloverTimeslice[smear_count]), std::ostream_iterator<double>(logstream, " "));
    }
    std::copy(std::cbegin(TopologicalChargeCloverTimeslice[n_smear]), std::prev(std::cend(TopologicalChargeCloverTimeslice[n_smear])), std::ostream_iterator<double>(logstream, " "));
    logstream << TopologicalChargeCloverTimeslice[n_smear].back() << "\n";
    //-----
    logstream << "TopChargePlaq: ";
    std::copy(std::cbegin(TopologicalChargePlaquette), std::prev(std::cend(TopologicalChargePlaquette)), std::ostream_iterator<double>(logstream, " "));
    logstream << TopologicalChargePlaquette.back() << "\n";
    //-----
    if (print_newline)
    {
        logstream << std::endl;
    }
}

void Observables(const GaugeField& Gluon, GaugeField& Gluonchain, std::ofstream& logstream, const MetaBiasPotential& Metapotential, const int n_count, const int n_smear, const double smearing_parameter = rho_stout)
{
    // Call the regular Observables() function, but do not print a newline at the end, since we still want to log the current CV
    Observables(Gluon, Gluonchain, logstream, n_count, n_smear, smearing_parameter, false);

    double CV_current {Metapotential.ReturnCV_current()};
    logstream << "CV_MetaD: " << CV_current << "\n";
    logstream << "Metapotential: " << Metapotential.ReturnPotential(CV_current) << "\n" << std::endl;
}

//-----

int main(int argc, char** argv)
{
    // iostream not synchronized with corresponding C streams, might cause a problem with C libraries and might not be thread safe
    std::ios_base::sync_with_stdio(false);
    std::cout << std::setprecision(12) << std::fixed;
    datalog << std::setprecision(12) << std::fixed;

    Configuration();
    CreateFiles();

    // Default width of random numbers used in Metropolis update is 0.5
    floatT         metropolis_epsilon    {0.5};
    constexpr int  n_therm               {20};
    constexpr bool accept_reject_enabled {true};
    if (!accept_reject_enabled)
    {
        std::cerr << Lettuce::Color::BoldRed << "Warning! Accept-reject step disabled!" << Lettuce::Color::Reset << std::endl;
    }

    // For rotating checkpoints
    int                        checkpoint_count      {0};
    std::array<std::string, 3> checkpoint_appendices {"_1", "_2", "_3"};

    Gluon.SetToIdentity();

    auto startcalc {std::chrono::system_clock::now()};
    datalog.open(logfilepath, std::fstream::out | std::fstream::app);

    // Commonly used gauge actions
    GaugeAction::WilsonAction.SetBeta(beta);
    GaugeAction::LüscherWeiszAction.SetBeta(beta);
    GaugeAction::IwasakiAction.SetBeta(beta);
    GaugeAction::DBW2Action.SetBeta(beta);

    // Initialize update functors
    HeatbathKernel                     Heatbath(Gluon, GaugeAction::DBW2Action, global_prng);
    // OverrelaxationDirectKernel         OverrelaxationDirect(Gluon, GaugeAction::WilsonAction, global_prng);
    OverrelaxationSubgroupKernel       OverrelaxationSubgroup(Gluon, GaugeAction::DBW2Action);
    Integrators::HMC::OMF_4            OMF_4_Integrator;
    // Integrators::HMC::Leapfrog_OMF_4   LFRG_OMF_4_Integrator;
    // Integrators::HMC::OMF_2_OMF_4      OMF_2_OMF_4_Integrator;
    GaugeUpdates::HMCKernel            HMC(Gluon, Gluonsmeared1, Gluonsmeared2, OMF_4_Integrator, GaugeAction::DBW2Action, global_prng, hmc_trajectory_length);
    // double ghmc_mixing_angle           {0.25 * pi<floatT>};
    // GaugeUpdates::GeneralizedHMCKernel GHMC(Gluon, Gluonsmeared1, GHMC_Momentum, Gluonsmeared2, OMF_4_Integrator, GaugeAction::WilsonAction, global_prng, ghmc_mixing_angle, hmc_trajectory_length);

    // LoadConfigBMW(Gluon, "GradientFlowBMW/conf0001.conf");

    // std::chrono::duration<double> overall_time {0.0};
    // Regular updates without Metadynamics
    if constexpr(!metadynamics_enabled)
    {
        // When using HMC, the thermalization is done without accept-reject step
        if constexpr(n_hmc != 0)
        {
            datalog << "[HMC start thermalization]\n";
            for (int n_count = 0; n_count < n_therm; ++n_count)
            {
                // auto start_therm_hmc {std::chrono::high_resolution_clock::now()};
                HMC(10, false);
                // auto end_therm_hmc {std::chrono::high_resolution_clock::now()};
                // std::chrono::duration<double> hmc_therm_time {end_therm_hmc - start_therm_hmc};
                // std::cout << "Time for thermalization sweep (HMC): " << hmc_therm_time.count() << std::endl;
            }
            datalog << "[HMC end thermalization]\n" << std::endl;
        }
        else
        {
            for (int n_count = 0; n_count < n_therm; ++n_count)
            {
                // auto start_therm {std::chrono::high_resolution_clock::now()};
                Iterator::Checkerboard4(Heatbath, n_heatbath);
                Iterator::Checkerboard4(OverrelaxationSubgroup, n_orelax);
                // auto end_therm {std::chrono::high_resolution_clock::now()};
                // std::chrono::duration<double> therm_time {end_therm - start_therm};
                // std::cout << "Time for thermalization sweep (local): " << therm_time.count() << std::endl;
            }
        }

        for (int n_count = 0; n_count < n_run; ++n_count)
        {
            if constexpr(n_metro != 0 and multi_hit != 0)
            {
                // auto start_update_metro {std::chrono::high_resolution_clock::now()};
                MetropolisKernel Metropolis(Gluon, GaugeAction::DBW2Action, global_prng, multi_hit, metropolis_epsilon);
                Iterator::Checkerboard4Sum(Metropolis, acceptance_count, n_metro);
                // TODO: Perhaps this should all happen automatically inside the functor?
                //       At the very least, we should probably combine the two actions below into one function
                Metropolis.AdjustEpsilon(acceptance_count);
                metropolis_epsilon = Metropolis.GetEpsilon();
                acceptance_count = 0;
                // auto end_update_metro {std::chrono::high_resolution_clock::now()};
                // std::chrono::duration<double> update_time_metro {end_update_metro - start_update_metro};
                // std::cout << "Time for " << n_metro << " Metropolis updates: " << update_time_metro.count() << std::endl;
            }
            //-----
            if constexpr(n_heatbath != 0)
            {
                // auto start_update_heatbath {std::chrono::high_resolution_clock::now()};
                Iterator::Checkerboard4(Heatbath, n_heatbath);
                // auto end_update_heatbath {std::chrono::high_resolution_clock::now()};
                // std::chrono::duration<double> update_time_heatbath {end_update_heatbath - start_update_heatbath};
                // std::cout << "Time for " << n_heatbath << " heatbath updates: " << update_time_heatbath.count() << std::endl;
                // overall_time += update_time_heatbath;
            }
            //-----
            if constexpr(n_hmc != 0)
            {
                // auto start_update_hmc {std::chrono::high_resolution_clock::now()};
                HMC(n_hmc, accept_reject_enabled);
                // auto end_update_hmc {std::chrono::high_resolution_clock::now()};
                // std::chrono::duration<double> update_time_hmc {end_update_hmc - start_update_hmc};
                // std::cout << "Time for one HMC trajectory: " << update_time_hmc.count() << std::endl;
                // overall_time += update_time_hmc;
            }
            //-----
            if constexpr(n_orelax != 0)
            {
                // auto start_update_or = std::chrono::high_resolution_clock::now();
                // double action_before {GaugeAction::WilsonAction.Action(Gluon)};
                // Iterator::CheckerboardSum(OverrelaxationDirect, acceptance_count_or, n_orelax);
                Iterator::Checkerboard4(OverrelaxationSubgroup, n_orelax);
                // double action_after {GaugeAction::WilsonAction.Action(Gluon)};
                // std::cout << "Action (before): " << action_before << std::endl;
                // std::cout << "Action (after): " << action_after << std::endl;
                // std::cout << action_after - action_before << std::endl;
                // auto end_update_or = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double> update_time_or {end_update_or - start_update_or};
                // std::cout << "Time for " << n_orelax << " OR updates: " << update_time_or.count() << std::endl;
                // overall_time += update_time_or;
            }
            //-----
            if constexpr(n_instanton_update != 0)
            {
                std::uniform_int_distribution<int> distribution_instanton(0, 1);
                int                                Q_instanton {distribution_instanton(generator_rand) * 2 - 1};
                int                                L_half      {Nt/2 - 1};
                site_coord                         center      {L_half, L_half, L_half, L_half};
                int                                radius      {5};
                // If the function is called for the first time, create Q = +1 and Q = -1 instanton configurations, otherwise reuse old configurations
                if (n_count == 0)
                {
                    BPSTInstantonUpdate(Gluon, Gluonsmeared1, Q_instanton, center, radius, acceptance_count_instanton, accept_reject_enabled, global_prng, true);
                }
                else
                {
                    BPSTInstantonUpdate(Gluon, Gluonsmeared1, Q_instanton, center, radius, acceptance_count_instanton, accept_reject_enabled, global_prng, false);
                }
            }
            //-----
            if (n_count % expectation_period == 0)
            {
                // auto start_observable = std::chrono::high_resolution_clock::now();
                Observables(Gluon, Gluonchain, datalog, n_count, n_smear, rho_stout);
                // auto end_observable = std::chrono::high_resolution_clock::now();
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
            if (n_count % checkpoint_period)
            {
                // Three rotating checkpoints, enable overwrite
                std::string checkpoint_appendix {checkpoint_appendices[checkpoint_count % 3]};
                SaveConfigBMW(Gluon, checkpointdirectory + "/config" + checkpoint_appendix + ".conf", true);
                global_prng.SaveState(checkpointdirectory + "/prng_state" + checkpoint_appendix + ".txt", checkpointdirectory + "/distribution_state" + checkpoint_appendix + ".txt", true);
            }
        }
    }

    // Updates with Metadynamics
    if constexpr(metadynamics_enabled and !tempering_enabled)
    {
        // CV_min, CV_max, bin_number, weight, well_tempered_parameter, threshold_weight
        // Original default values
        // MetaBiasPotential TopBiasPotential{-8, 8, 800, 0.05, 100, 1000.0};
        // New attempt at values for well tempered updates
        MetaBiasPotential TopBiasPotential{-8, 8, 800, 0.1, 50, 1000.0};
        // TopBiasPotential.GeneratePotentialFrom([](double CV_in){return std::fmax(-0.25 * CV_in * CV_in - 14.0 * std::pow(std::sin(1.2 * pi<floatT> * CV_in), 2) + 43.0, 0.0);});
        // TopBiasPotential.LoadPotential("SU(3)_N=20x20x20x20_beta=1.250000/metapotential.txt");
        // TopBiasPotential.SymmetrizePotential();
        // TopBiasPotential.SymmetrizePotentialMaximum();
        TopBiasPotential.SaveParameters(metapotentialfilepath);
        TopBiasPotential.SavePotential(metapotentialfilepath);

        // GaugeUpdates::HMCMetaDKernel HMC_MetaD(Gluon, Gluonsmeared1, Gluonsmeared2, TopBiasPotential, OMF_2_OMF_4_Integrator, GaugeAction::DBW2Action, n_smear_meta, global_prng, hmc_trajectory_length, rho_stout_metadynamics);
        GaugeUpdates::HMCMetaDData   MetadynamicsData(n_smear_meta);
        GaugeUpdates::HMCMetaDKernel HMC_MetaD(Gluon, Gluonsmeared1, Gluonsmeared2, TopBiasPotential, MetadynamicsData, OMF_4_Integrator, GaugeAction::DBW2Action, global_prng, hmc_trajectory_length, rho_stout_metadynamics);

        // Thermalize with normal HMC
        datalog << "[HMC start thermalization]\n";
        for (int n_count = 0; n_count < n_therm; ++n_count)
        {
            // Iterator::Checkerboard(Heatbath, 1);
            // Iterator::Checkerboard(OverrelaxationSubgroup, 4);
            HMC(10, false);
        }
        datalog << "[HMC end thermalization]\n" << std::endl;

        for (int n_count = 0; n_count < n_run; ++n_count)
        {
            // auto start_update_meta = std::chrono::high_resolution_clock::now();
            HMC_MetaD(n_hmc, accept_reject_enabled);
            // auto end_update_meta = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> update_time_meta {end_update_meta - start_update_meta};
            // std::cout << "Time for meta update: " << update_time_meta.count() << std::endl;
            // overall_time += update_time_meta;
            if (n_count % expectation_period == 0)
            {
                Observables(Gluon, Gluonchain, datalog, TopBiasPotential, n_count, n_smear, rho_stout);
                if constexpr(metapotential_updated)
                {
                    if (n_count % (1 * expectation_period) == 0)
                    TopBiasPotential.SavePotential(metapotentialfilepath);
                }
            }
            if (n_count % checkpoint_period)
            {
                // Three rotating checkpoints, enable overwrite
                std::string checkpoint_appendix {checkpoint_appendices[checkpoint_count % 3]};
                SaveConfigBMW(Gluon, checkpointdirectory + "/config" + checkpoint_appendix + ".conf", true);
                global_prng.SaveState(checkpointdirectory + "/prng_state" + checkpoint_appendix + ".txt", checkpointdirectory + "/distribution_state" + checkpoint_appendix + ".txt", true);
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

        Gluon_temper.SetToIdentity();

        // Setup second ofstream for Gluon_temper (Gluon uses the default stream datalog)
        std::ofstream datalog_temper;
        datalog_temper << std::setprecision(12) << std::fixed;
        std::string logfilepath_temper = directoryname + "/log_temper.txt";
        datalog_temper.open(logfilepath_temper, std::fstream::out | std::fstream::app);

        // Conventional HMC only used during thermalization of Gluon_temper
        GaugeUpdates::HMCKernel                   HMC_temper(Gluon_temper, Gluonsmeared1, Gluonsmeared2, OMF_4_Integrator, GaugeAction::DBW2Action, global_prng, hmc_trajectory_length);

        // CV_min, CV_max, bin_number, weight, well_tempered_parameter, threshold_weight
        // Original default values
        // MetaBiasPotential                         TopBiasPotential{-8, 8, 800, 0.05, 100, 1000.0};
        // New attempt at values for well tempered updates
        MetaBiasPotential                         TopBiasPotential{-8, 8, 800, 0.1, 50, 1000.0};
        // TopBiasPotential.LoadPotential("metapotential_16_1.24.txt");
        // TopBiasPotential.SymmetrizePotentialMaximum();
        TopBiasPotential.SaveParameters(metapotentialfilepath);
        TopBiasPotential.SavePotential(metapotentialfilepath);
        GaugeUpdates::HMCMetaDData                MetadynamicsData(n_smear_meta);
        GaugeUpdates::HMCMetaDKernel              HMC_MetaD(Gluon_temper, Gluonsmeared1, Gluonsmeared2, TopBiasPotential, MetadynamicsData, OMF_4_Integrator, GaugeAction::DBW2Action, global_prng, hmc_trajectory_length, rho_stout_metadynamics);

        GaugeUpdates::MetadynamicsTemperingKernel ParallelTemperingSwap(Gluon, Gluon_temper, Gluonsmeared1, Gluonsmeared2, TopBiasPotential, global_prng, rho_stout_metadynamics);

        // Thermalize Gluon with local updates, and Gluon_temper with normal HMC
        datalog << "[HMC start thermalization]\n";
        for (int n_count = 0; n_count < n_therm; ++n_count)
        {
            Iterator::Checkerboard4(Heatbath, n_heatbath);
            Iterator::Checkerboard4(OverrelaxationSubgroup, n_orelax);
            HMC_temper(10, false);
        }
        datalog << "[HMC end thermalization]\n" << std::endl;

        for (int n_count = 0; n_count < n_run; ++n_count)
        {
            // Perform updates on Gluon (for every MetaD-HMC update perform tempering_nonmetadynamics_sweeps updates on the config without Metadynamics)
            for (int n_count_nobias = 0; n_count_nobias < tempering_nonmetadynamics_sweeps; ++n_count_nobias)
            {
                Iterator::Checkerboard4(Heatbath, n_heatbath);
                Iterator::Checkerboard4(OverrelaxationSubgroup, n_orelax);
            }

            // Perform updates on Gluon_temper
            HMC_MetaD(n_hmc, accept_reject_enabled);

            // Propose tempering swap
            if (n_count % tempering_swap_period == 0)
            {
                datalog_temper << "Tempering swap accepted: " << ParallelTemperingSwap() << std::endl;
            }

            if (n_count % expectation_period == 0)
            {
                Observables(Gluon, Gluonchain, datalog, n_count, n_smear, rho_stout);
                Observables(Gluon_temper, Gluonchain, datalog_temper, TopBiasPotential, n_count, n_smear, rho_stout);
                if constexpr(metapotential_updated)
                {
                    if (n_count % (1 * expectation_period) == 0)
                    TopBiasPotential.SavePotential(metapotentialfilepath);
                }
            }
            if (n_count % checkpoint_period)
            {
                // Three rotating checkpoints, enable overwrite
                std::string checkpoint_appendix {checkpoint_appendices[checkpoint_count % 3]};
                SaveConfigBMW(Gluon, checkpointdirectory + "/config" + checkpoint_appendix + ".conf", true);
                SaveConfigBMW(Gluon_temper, checkpointdirectory + "/config_temper" + checkpoint_appendix + ".conf", true);
                global_prng.SaveState(checkpointdirectory + "/prng_state" + checkpoint_appendix + ".txt", checkpointdirectory + "/distribution_state" + checkpoint_appendix + ".txt", true);
            }
        }
        datalog_temper.close();
        datalog_temper.clear();
    }

    auto end {std::chrono::system_clock::now()};
    std::chrono::duration<double> elapsed_seconds {end - startcalc};
    std::time_t end_time {std::chrono::system_clock::to_time_t(end)};

    // std::cout << "Overall time:    " << overall_time.count() << std::endl;
    // std::cout << "Normalized time: " << overall_time.count() / n_run << std::endl;

    //-----
    // Save final configuration and PRNG state
    SaveConfigBMW(Gluon, checkpointdirectory + "/final_config.conf");
    global_prng.SaveState(checkpointdirectory + "/prng_state.txt", checkpointdirectory + "/distribution_state.txt");

    // Print acceptance rates, PRNG width, and required time to terminal and to files

    std::cout << "\n";
    PrintFinal(std::cout, acceptance_count, acceptance_count_or, acceptance_count_hmc, acceptance_count_metadynamics_hmc, acceptance_count_tempering, metropolis_epsilon, end_time, elapsed_seconds);

    PrintFinal(datalog, acceptance_count, acceptance_count_or, acceptance_count_hmc, acceptance_count_metadynamics_hmc, acceptance_count_tempering, metropolis_epsilon, end_time, elapsed_seconds);
    datalog.close();
    datalog.clear();

    datalog.open(parameterfilepath, std::fstream::out | std::fstream::app);
    PrintFinal(datalog, acceptance_count, acceptance_count_or, acceptance_count_hmc, acceptance_count_metadynamics_hmc, acceptance_count_tempering, metropolis_epsilon, end_time, elapsed_seconds);
    datalog.close();
    datalog.clear();
}
