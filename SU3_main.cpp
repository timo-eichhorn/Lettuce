// Pure SU(3) theory
// Command line flags: -DFIXED_SEED

// Non-standard library headers
// Include these three header files first in this order
// TODO: Should probably check all includes and add the appropriate includes to all files
#include "LettuceGauge/defines.hpp"
#include "LettuceGauge/coords.hpp"
#include "LettuceGauge/lattice.hpp"
//-----
// Remaining files in alphabetic order (for now)
#include "LettuceGauge/actions/bias_potential/metadynamics.hpp"
#include "LettuceGauge/actions/bias_potential/variational_bias.hpp"
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
#include "LettuceGauge/updates/parity_update.hpp"
#include "LettuceGauge/updates/tempering.hpp"
#include "LettuceGauge/utility/timer.hpp"
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

GaugeField                   Gluon;
GaugeField                   Gluonsmeared1;
GaugeField                   Gluonsmeared2;
GaugeField                   Gluonchain;
FullTensor                   F_tensor;

//-------------------------------------------------------------------------------------
// Calculates and writes observables to logfile

void SaveObservable(const std::vector<double>& observable_vec, const std::string& observable_name, std::ofstream& logstream)
{
    logstream << observable_name;
    std::copy(std::cbegin(observable_vec), std::prev(std::cend(observable_vec)), std::ostream_iterator<double>(logstream, " "));
    logstream << observable_vec.back() << "\n";
}

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
    // Timer action_timer;
    // Action[0]                      = WilsonAction::ActionNormalized(Gluon);
    // std::cout << "Time for calculating action: " << action_timer.GetTimeSeconds() << std::endl;
    ActionImproved[0]              = SymanzikAction.ActionNormalized(Gluon);
    Plaquette[0]                   = PlaquetteSum(Gluon);

    FieldStrengthTensor::Clover(Gluon, F_tensor);
    // Timer topcharge_timeslice_timer;
    for (int t = 0; t < Gluon.Length(0); ++t)
    {
        TopologicalChargeCloverTimeslice[0][t] = TopChargeCloverTimeslice(F_tensor, t);
    }
    // std::cout << "Time for calculating topcharge timeslice: " << topcharge_timeslice_timer.GetTimeSeconds() << "\n";
    // Timer topcharge_symm_timer;
    TopologicalChargeClover[0]     = std::accumulate(TopologicalChargeCloverTimeslice[0].cbegin(), TopologicalChargeCloverTimeslice[0].cend(), 0.0);
    // TopologicalChargeClover[0]     = TopChargeClover(Gluon);
    // TopologicalChargeClover[0]     = TopologicalCharge::CloverChargeFromFTensor(F_tensor);
    // std::cout << "Time for calculating topcharge (symm): " << topcharge_symm_timer.GetTimeSeconds() << "\n";
    // Timer topcharge_plaq_timer;
    TopologicalChargePlaquette[0]  = TopChargePlaquette(Gluon);
    // std::cout << "Time for calculating topcharge (plaq): " << topcharge_plaq_timer.GetTimeSeconds() << "\n";

    FieldStrengthTensor::MakeComponentsTraceless(F_tensor);
    for (int t = 0; t < Gluon.Length(0); ++t)
    {
        ECloverTimeslice[0][t]     = EnergyDensity::CloverTimeslice(F_tensor, t);
        // EPlaquetteTimeslice[0][t]  = EnergyDensity::PlaquetteTimeslice(Gluon, t);
    }
    // EClover[0]                     = EnergyDensity::Clover(F_tensor);
    EClover[0]                     = std::accumulate(ECloverTimeslice[0].cbegin(), ECloverTimeslice[0].cend(), 0.0);

    // Timer wilson_timer;
    WLoop2[0]                      = WilsonLoop<0, 2,  true>(Gluon, Gluonchain);
    // std::cout << "Time for calculating wilson 2: " << wilson_timer.GetTimeSeconds() << "\n";

    // wilson_timer.Reset();
    WLoop4[0]                      = WilsonLoop<2, 4, false>(Gluon, Gluonchain);
    // std::cout << "Time for calculating wilson 4: " << wilson_timer.GetTimeSeconds() << "\n";

    // wilson_timer.Reset();
    WLoop8[0]                      = WilsonLoop<4, 8, false>(Gluon, Gluonchain);
    // std::cout << "Time for calculating wilson 8: " << wilson_timer.GetTimeSeconds() << "\n";

    // Timer polyakov_timer;
    PLoop[0]                       = PolyakovLoop(Gluon);
    // std::cout << "Time for calculating Polyakov: " << polyakov_timer.GetTimeSeconds() << "\n";

    //-----
    // Measurements involving smearing
    for (int smear_count = 1; smear_count <= n_smear; ++smear_count)
    {
        // Apply smearing (first call is distinct from the calls afterwards, since we need to copy the unsmeared gaugefield here, but not later on)
        if (smear_count == 1)
        {
            // Timer smear_timer;
            Flow(n_smear_skip);
            // std::cout << "Time for smearing with " << Flow.ReturnIntegratorName() << ": " << smear_timer.GetTimeSeconds() << "\n";
        }
        else
        {
            Flow.Resume(n_smear_skip);
        }
        // Iterator::Checkerboard(Cooling, n_smear_skip);
        // Calculate observables
        ActionImproved[smear_count]              = SymanzikAction.ActionNormalized(Gluonsmeared1);
        Plaquette[smear_count]                   = PlaquetteSum(Gluonsmeared1);
        FieldStrengthTensor::Clover(Gluonsmeared1, F_tensor);
        for (int t = 0; t < Gluon.Length(0); ++t)
        {
            TopologicalChargeCloverTimeslice[smear_count][t] = TopChargeCloverTimeslice(F_tensor, t);
        }
        TopologicalChargeClover[smear_count]     = std::accumulate(TopologicalChargeCloverTimeslice[smear_count].cbegin(), TopologicalChargeCloverTimeslice[smear_count].cend(), 0.0);
        // TopologicalChargeClover[smear_count]     = TopChargeClover(Gluonsmeared1);
        // TopologicalChargeClover[smear_count]     = TopologicalCharge::CloverChargeFromFTensor(F_tensor);
        TopologicalChargePlaquette[smear_count]  = TopChargePlaquette(Gluonsmeared1);
        FieldStrengthTensor::MakeComponentsTraceless(F_tensor);
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
    //----
    SaveObservable(Action, "Wilson_Action: ", logstream);
    SaveObservable(ActionImproved, "Improved_Action: ", logstream);
    SaveObservable(Plaquette, "Plaquette: ", logstream);
    SaveObservable(EPlaquette, "E_plaq: ", logstream);
    // logstream << "E_plaq_timeslice: ";
    // for (int smear_count = 0; smear_count < n_smear; ++smear_count)
    // {
    //     std::copy(std::cbegin(EPlaquetteTimeslice[smear_count]), std::cend(EPlaquetteTimeslice[smear_count]), std::ostream_iterator<double>(logstream, " "));
    // }
    // std::copy(std::cbegin(EPlaquetteTimeslice[n_smear]), std::prev(std::cend(EPlaquetteTimeslice[n_smear])), std::ostream_iterator<double>(logstream, " "));
    // logstream << EPlaquetteTimeslice[n_smear].back() << "\n";
    SaveObservable(EClover, "E_clov: ", logstream);
    logstream << "E_clov_timeslice: ";
    for (int smear_count = 0; smear_count < n_smear; ++smear_count)
    {
        std::copy(std::cbegin(ECloverTimeslice[smear_count]), std::cend(ECloverTimeslice[smear_count]), std::ostream_iterator<double>(logstream, " "));
    }
    std::copy(std::cbegin(ECloverTimeslice[n_smear]), std::prev(std::cend(ECloverTimeslice[n_smear])), std::ostream_iterator<double>(logstream, " "));
    logstream << ECloverTimeslice[n_smear].back() << "\n";
    SaveObservable(ActionUnnormalized, "Wilson_Action(unnormalized): ", logstream);
    SaveObservable(WLoop2, "Wilson_loop(L=2): ", logstream);
    SaveObservable(WLoop4, "Wilson_loop(L=4): ", logstream);
    SaveObservable(WLoop8, "Wilson_loop(L=8): ", logstream);
    SaveObservable(PLoopRe, "Polyakov_loop(Re): ", logstream);
    SaveObservable(PLoopIm, "Polyakov_loop(Im): ", logstream);
    SaveObservable(TopologicalChargeClover, "TopChargeClov: ", logstream);
    logstream << "TopChargeClov_timeslice: ";
    for (int smear_count = 0; smear_count < n_smear; ++smear_count)
    {
        std::copy(std::cbegin(TopologicalChargeCloverTimeslice[smear_count]), std::cend(TopologicalChargeCloverTimeslice[smear_count]), std::ostream_iterator<double>(logstream, " "));
    }
    std::copy(std::cbegin(TopologicalChargeCloverTimeslice[n_smear]), std::prev(std::cend(TopologicalChargeCloverTimeslice[n_smear])), std::ostream_iterator<double>(logstream, " "));
    logstream << TopologicalChargeCloverTimeslice[n_smear].back() << "\n";
    SaveObservable(TopologicalChargePlaquette, "TopChargePlaq: ", logstream);
    //-----
    if (print_newline)
    {
        logstream << std::endl;
    }
}

template<typename BiasPotentialT>
void Observables(const GaugeField& Gluon, GaugeField& Gluonchain, std::ofstream& logstream, const BiasPotentialT& Metapotential, const int n_count, const int n_smear, const double smearing_parameter = rho_stout)
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
    datalog   << std::setprecision(12) << std::fixed;

    // Get command line arguments in reversed order (that way if a parameter is found multiple times, only the last appearance is considered)
    std::vector<std::string> command_line_arguments(argv, argv + argc);
    std::reverse(command_line_arguments.begin(), command_line_arguments.end());

    Configuration(command_line_arguments);
    CreateFiles();

    // omp_set_schedule(omp_sched_static);

    // Default width of random numbers used in Metropolis of 0.5 probably not optimal and parameter-dependent, so tune on case-by-case basis
    floatT         metropolis_epsilon    {0.5};
    constexpr bool accept_reject_enabled {true};
    if (!accept_reject_enabled)
    {
        std::cerr << Lettuce::Color::BoldRed << "Warning! Accept-reject step disabled!" << Lettuce::Color::Reset << std::endl;
    }

    // For rotating checkpoints
    constexpr bool    create_checkpoint_subdirectories {true};
    CheckpointManager Checkpointer(checkpointdirectory, n_checkpoint_backups, create_checkpoint_subdirectories);
    std::cout << "Automatic checkpoint directory in " << Checkpointer.CheckpointDirectory() << std::endl;

    Gluon.SetToIdentity();
    if (extend_run)
    {
        if (std::filesystem::exists(old_maindirectory + "/checkpoints/final_config.conf")
        and std::filesystem::exists(old_maindirectory + "/checkpoints/final_prng_state.txt")
        and std::filesystem::exists(old_maindirectory + "/checkpoints/final_distribution_state.txt"))
        {
            LoadConfigBMWFull(Gluon, old_maindirectory + "/checkpoints/final_config.conf");
            global_prng.LoadState(old_maindirectory + "/checkpoints/final_prng_state.txt", old_maindirectory + "/checkpoints/final_distribution_state.txt");
        }
        else if (std::filesystem::exists(old_maindirectory + "/checkpoints/config.conf")
             and std::filesystem::exists(old_maindirectory + "/checkpoints/prng_state.txt")
             and std::filesystem::exists(old_maindirectory + "/checkpoints/distribution_state.txt"))
        {
            LoadConfigBMWFull(Gluon, old_maindirectory + "/checkpoints/config.conf");
            global_prng.LoadState(old_maindirectory + "/checkpoints/prng_state.txt", old_maindirectory + "/checkpoints/distribution_state.txt");
        }
    }

    auto startcalc {std::chrono::system_clock::now()};
    datalog.open(logfilepath, std::fstream::out | std::fstream::app);

    // Commonly used gauge actions
    GaugeAction::WilsonAction.SetBeta(beta);
    GaugeAction::LüscherWeiszAction.SetBeta(beta);
    GaugeAction::IwasakiAction.SetBeta(beta);
    GaugeAction::DBW2Action.SetBeta(beta);

    auto SimulatedAction = GaugeAction::WilsonAction;
    std::cout << "SimulatedAction parameters: " << SimulatedAction.stencil_radius << ", " << SimulatedAction.c_plaq << ", " << SimulatedAction.c_rect << std::endl;

    // Initialize update functors
    HeatbathKernel                     Heatbath(Gluon, SimulatedAction, global_prng);
    // OverrelaxationDirectKernel         OverrelaxationDirect(Gluon, SimulatedAction, global_prng);
    OverrelaxationSubgroupKernel       OverrelaxationSubgroup(Gluon, SimulatedAction);
    using HMC_IntegratorT = Integrators::HMC::OMF_4;
    HMC_IntegratorT                    HMC_Integrator;
    GaugeUpdates::HMCKernel            HMC(Gluon, Gluonsmeared1, Gluonsmeared2, HMC_Integrator, SimulatedAction, global_prng, hmc_trajectory_length);
    // double ghmc_mixing_angle           {0.25 * pi<floatT>};
    // GaugeUpdates::GeneralizedHMCKernel GHMC(Gluon, Gluonsmeared1, GHMC_Momentum, Gluonsmeared2, HMC_Integrator, SimulatedAction, global_prng, ghmc_mixing_angle, hmc_trajectory_length);

    // LoadConfigBMW(Gluon, "GradientFlowBMW/conf0001.conf");

    // std::chrono::duration<double> overall_time {0.0};
    Timer overall_timer;
    // Regular updates without Metadynamics
    if constexpr(!metadynamics_enabled)
    {
        // When using HMC, the thermalization is done without accept-reject step
        if constexpr(n_hmc != 0)
        {
            datalog << "[HMC start thermalization]\n";
            for (int n_count = 0; n_count < n_therm; ++n_count)
            {
                // Timer hmc_thermalization_timer;
                HMC(2 * n_hmc, false);
                // std::cout << "Time for thermalization sweep (HMC): " << hmc_thermalization_timer.GetTimeSeconds() << "\n";
            }
            datalog << "[HMC end thermalization]\n" << std::endl;
        }
        else
        {
            for (int n_count = 0; n_count < n_therm; ++n_count)
            {
                // Timer thermalization_timer;
                Iterator::Checkerboard4(Heatbath, n_heatbath);
                Iterator::Checkerboard4(OverrelaxationSubgroup, n_orelax);
                // std::cout << "Time for thermalization sweep (local): " << thermalization_timer.GetTimeSeconds() << "\n";
            }
        }

        for (int n_count = 0; n_count < n_run; ++n_count)
        {
            if constexpr(n_metro != 0 and multi_hit != 0)
            {
                // Timer metropolis_timer;
                MetropolisKernel Metropolis(Gluon, SimulatedAction, global_prng, multi_hit, metropolis_epsilon);
                Iterator::Checkerboard4Sum(Metropolis, acceptance_count, n_metro);
                // TODO: Perhaps this should all happen automatically inside the functor?
                //       At the very least, we should probably combine the two actions below into one function
                // Metropolis.AdjustEpsilon(acceptance_count);
                metropolis_epsilon = Metropolis.GetEpsilon();
                acceptance_count = 0;
                // std::cout << "Time for " << n_metro << " Metropolis updates: " << metropolis_timer.GetTimeSeconds() << "\n";
            }
            //-----
            if constexpr(n_heatbath != 0)
            {
                // Timer heatbath_timer;
                Iterator::Checkerboard4(Heatbath, n_heatbath);
                // std::cout << "Time for " << n_heatbath << " heatbath updates: " << heatbath_timer.GetTimeSeconds() << "\n";
            }
            //-----
            if constexpr(n_hmc != 0)
            {
                // Timer hmc_timer;
                HMC(n_hmc, accept_reject_enabled);
                // std::cout << "Time for one HMC trajectory: " << hmc_timer.GetTimeSeconds() << "\n";
            }
            //-----
            if constexpr(n_orelax != 0)
            {
                // Timer overrelaxation_timer;
                // double action_before {SimulatedAction.Action(Gluon)};
                // Iterator::CheckerboardSum(OverrelaxationDirect, acceptance_count_or, n_orelax);
                Iterator::Checkerboard4(OverrelaxationSubgroup, n_orelax);
                // double action_after {SimulatedAction.Action(Gluon)};
                // std::cout << "Action (before): " << action_before << std::endl;
                // std::cout << "Action (after): " << action_after << std::endl;
                // std::cout << action_after - action_before << std::endl;
                // std::cout << "Time for " << n_orelax << " OR updates: " << overrelaxation_timer.GetTimeSeconds() << "\n";
            }
            //-----
            if constexpr(n_instanton_update != 0)
            {
                std::uniform_int_distribution<int>  distribution_instanton(0, 1);
                int                                 Q_instanton {distribution_instanton(generator_rand) * 2 - 1};
                int                                 L_half      {Nt/2 - 1};
                site_coord                          center      {L_half, L_half, L_half, L_half};
                int                                 radius      {5};
                GaugeUpdates::InstantonUpdateKernel InstantonUpdate(Gluon, Gluonsmeared1, SimulatedAction, global_prng, center, radius, false);
                // If the function is called for the first time, create Q = +1 and Q = -1 instanton configurations, otherwise reuse old configurations
                if (n_count == 0)
                {
                    // BPSTInstantonUpdate(Gluon, Gluonsmeared1, Q_instanton, center, radius, acceptance_count_instanton, accept_reject_enabled, global_prng, true);
                    InstantonUpdate.CreateBPSTInstantons(center, radius);
                    InstantonUpdate(Q_instanton, accept_reject_enabled);
                }
                else
                {
                    // BPSTInstantonUpdate(Gluon, Gluonsmeared1, Q_instanton, center, radius, acceptance_count_instanton, accept_reject_enabled, global_prng, false);
                    InstantonUpdate(Q_instanton, accept_reject_enabled);
                }
            }
            //-----
            if (n_count % expectation_period == 0)
            {
                // Timer observables_timer;
                Observables(Gluon, Gluonchain, datalog, n_count, n_smear, rho_stout);
                // std::cout << "Time for calculating observables: " << observables_timer.GetTimeSeconds() << std::endl;

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
            if (n_count % checkpoint_period == 0)
            {
                Checkpointer.AlternatingCheckpoints(SaveConfigBMWFull, global_prng, Gluon, "config.conf", "prng_state.txt", "distribution_state.txt");
            }
        }
    }

    // Updates with Metadynamics
    if constexpr(metadynamics_enabled and !tempering_enabled)
    {
        // With uniform distribution the initial range has to be chosen much smaller
                         // double Q_min_initial      = -0.2;
                         // double Q_max_initial      =  0.2;
        [[maybe_unused]] double Q_min_initial           = -0.6;
        [[maybe_unused]] double Q_max_initial           =  0.6;
                         double Q_min                   = -8.0;
                         double Q_max                   =  8.0;
        [[maybe_unused]] double ves_stepsize            =  0.5;
        [[maybe_unused]] double ves_momentum            =  0.9; // Seems to be the default value for Polyak momentum
        [[maybe_unused]] int    bin_number              =  800;
        [[maybe_unused]] double gaussian_height         =  metapotential_well_tempered ? 0.05 : 0.025;
        [[maybe_unused]] double gaussian_width          =  4 * std::abs(Q_max - Q_min) / bin_number;
        [[maybe_unused]] double well_tempered_parameter =  10.0;
        [[maybe_unused]] double threshold_weight        =  1000.0;

        // Metadynamics
        MetaBiasPotential TopBiasPotential{Q_min, Q_max, bin_number, gaussian_height, gaussian_width, well_tempered_parameter, threshold_weight};

        // TopBiasPotential.GeneratePotentialFrom([](double CV_in){return std::fmax(-0.25 * CV_in * CV_in - 14.0 * std::pow(std::sin(1.2 * pi<floatT> * CV_in), 2) + 43.0, 0.0);});
        // TopBiasPotential.LoadPotential("SU(3)_N=20x20x20x20_beta=1.250000/metapotential.txt");
        // TopBiasPotential.SymmetrizePotential();
        // TopBiasPotential.SymmetrizePotentialMaximum();

        // // Variationally enhanced sampling
        // int ves_initial_batchsize = 50 * n_hmc;

        // using VESParametersT = SimpleBasis::ParametersT;
        // Optimizers::AveragedStochasticGradientDescent<VESParametersT> sgd_optimizer(ves_stepsize, ves_momentum, ves_initial_batchsize);
        // // Optimizers::Adam<VESParametersT>                              adam_optimizer(ves_initial_batchsize);
        // // VariationalBiasPotential TopBiasPotential(SimpleBasis{0.0, 0.0}, UniformTargetDistribution{}, sgd_optimizer, Q_min, Q_max, Q_min_initial, Q_max_initial, ves_initial_batchsize);
        // VariationalBiasPotential TopBiasPotential{SimpleBasis{0.0, 0.0}, GaussianTargetDistribution{0.0, 4}, sgd_optimizer, Q_min, Q_max, Q_min_initial, Q_max_initial, ves_initial_batchsize};

        TopBiasPotential.SaveParameters(metapotentialfilepath);
        TopBiasPotential.SavePotential(metapotentialfilepath);

        GaugeUpdates::HMCMetaDData   MetadynamicsData(n_smear_meta);
        GaugeUpdates::HMCMetaDKernel HMC_MetaD(Gluon, Gluonsmeared1, Gluonsmeared2, TopBiasPotential, MetadynamicsData, HMC_Integrator, SimulatedAction, global_prng, hmc_trajectory_length, rho_stout_metadynamics);

        // GaugeUpdates::InstantonStart(Gluon, 1);

        std::uniform_int_distribution<int>  distribution_parity_update(0, 1);

        // Thermalize with normal HMC
        datalog << "[HMC start thermalization]\n";
        for (int n_count = 0; n_count < n_therm; ++n_count)
        {
            // Iterator::Checkerboard(Heatbath, 1);
            // Iterator::Checkerboard(OverrelaxationSubgroup, 4);
            HMC(2 * n_hmc, false);
        }
        datalog << "[HMC end thermalization]\n" << std::endl;

        for (int n_count = 0; n_count < n_run; ++n_count)
        {
            // Timer metad_timer;
            HMC_MetaD(n_hmc, accept_reject_enabled);
            // std::cout << "Time for meta update: " << metad_timer.GetTimeSeconds() << "\n";
            if (distribution_parity_update(generator_rand))
            {
                ParityUpdate(Gluon, Gluonsmeared1);
            }
            if (n_count % expectation_period == 0)
            {
                Observables(Gluon, Gluonchain, datalog, TopBiasPotential, n_count, n_smear, rho_stout);
                if constexpr(metapotential_update_stride >= 1)
                {
                    if (n_count % (1 * expectation_period) == 0)
                    TopBiasPotential.SavePotential(metapotentialfilepath);
                }
            }
            if (n_count % checkpoint_period == 0)
            {
                Checkpointer.AlternatingCheckpoints(SaveConfigBMWFull, global_prng, Gluon, "config.conf", "prng_state.txt", "distribution_state.txt");
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
        datalog_temper.open(logfilepath_temper, std::fstream::out | std::fstream::app);

        // Conventional HMC only used during thermalization of Gluon_temper
        GaugeUpdates::HMCKernel                   HMC_temper(Gluon_temper, Gluonsmeared1, Gluonsmeared2, HMC_Integrator, SimulatedAction, global_prng, hmc_trajectory_length);

        // [[maybe_unused]] double Q_min_initial           = -0.6;
        // [[maybe_unused]] double Q_max_initial           =  0.6;
                         double Q_min                   = -8.0;
                         double Q_max                   =  8.0;
        // [[maybe_unused]] double ves_stepsize            =  0.5;
        // [[maybe_unused]] double ves_momentum            =  0.9; // Seems to be the default value for Polyak momentum
        /*[[maybe_unused]]*/ int    bin_number              =  800;
        /*[[maybe_unused]]*/ double gaussian_height         =  metapotential_well_tempered ? 0.05 : 0.025;
        /*[[maybe_unused]]*/ double gaussian_width          =  4 * std::abs(Q_max - Q_min) / bin_number;
        /*[[maybe_unused]]*/ double well_tempered_parameter =  10.0;
        /*[[maybe_unused]]*/ double threshold_weight        =  1000.0;

        MetaBiasPotential                         TopBiasPotential{Q_min, Q_max, bin_number, gaussian_height, gaussian_width, well_tempered_parameter, threshold_weight};

        // TopBiasPotential.LoadPotential("metapotential_16_1.24.txt");
        // TopBiasPotential.SymmetrizePotentialMaximum();

        TopBiasPotential.SaveParameters(metapotentialfilepath);
        TopBiasPotential.SavePotential(metapotentialfilepath);

        GaugeUpdates::HMCMetaDData                MetadynamicsData(n_smear_meta);
        GaugeUpdates::HMCMetaDKernel              HMC_MetaD(Gluon_temper, Gluonsmeared1, Gluonsmeared2, TopBiasPotential, MetadynamicsData, HMC_Integrator, SimulatedAction, global_prng, hmc_trajectory_length, rho_stout_metadynamics);
        GaugeUpdates::MetadynamicsTemperingKernel ParallelTemperingSwap(Gluon, Gluon_temper, Gluonsmeared1, Gluonsmeared2, TopBiasPotential, global_prng, rho_stout_metadynamics);

        // Thermalize Gluon with local updates, and Gluon_temper with normal HMC
        datalog << "[HMC start thermalization]\n";
        for (int n_count = 0; n_count < n_therm; ++n_count)
        {
            Iterator::Checkerboard4(Heatbath, n_heatbath);
            Iterator::Checkerboard4(OverrelaxationSubgroup, n_orelax);
            HMC_temper(2 * n_hmc, false);
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
                if constexpr(metapotential_update_stride >= 1)
                {
                    if (n_count % (1 * expectation_period) == 0)
                    TopBiasPotential.SavePotential(metapotentialfilepath);
                }
            }
            if (n_count % checkpoint_period == 0)
            {
                Checkpointer.AlternatingCheckpoints(SaveConfigBMWFull, global_prng, Gluon, "config.conf", "prng_state.txt", "distribution_state.txt");
                Checkpointer.AlternatingConfigCheckpoints(SaveConfigBMWFull, Gluon_temper, "config_temper.conf");
            }
        }
        datalog_temper.close();
        datalog_temper.clear();
    }

    auto end {std::chrono::system_clock::now()};
    std::chrono::duration<double> elapsed_seconds {end - startcalc};
    std::time_t end_time {std::chrono::system_clock::to_time_t(end)};

    std::cout << "Overall time:    " << overall_timer.GetTimeSeconds() << std::endl;
    std::cout << "Normalized time: " << overall_timer.GetTimeSeconds() / n_run << std::endl;

    //-----
    // Save final configuration and PRNG state
    SaveConfigBMWFull(Gluon, checkpointdirectory + "/final_config.conf");
    global_prng.SaveState(checkpointdirectory + "/final_prng_state.txt", checkpointdirectory + "/final_distribution_state.txt");

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
    Checkpointer.CreateCompletedFile(maindirectory);
}
