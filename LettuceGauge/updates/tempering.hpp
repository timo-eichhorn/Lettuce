#ifndef LETTUCE_TEMPERING_HPP
#define LETTUCE_TEMPERING_HPP

// Non-standard library headers
#include "../defines.hpp"
// #include "../lattice.hpp"
#include "../metadynamics.hpp"
#include "../observables/topological_charge.hpp"
#include "../smearing/gradient_flow.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <random>
//----------------------------------------
// Standard C headers
#include <cmath>

namespace GaugeUpdates
{
    template<typename prngT>
    struct MetadynamicsTemperingKernel
    {
        private:
            GaugeField&        U;
            GaugeField&        U_temper;
            // Fields we need temporarily during smearing
            GaugeField&        U_copy1;
            GaugeField&        U_copy2;
            MetaBiasPotential& Metapotential;
            prngT&             prng;
        public:
            double             rho_stout_cv;

            double MetaCharge(const GaugeField& Gluon, GaugeField& Gluon_smeared, GaugeField& Forcefield, const int n_smear_meta, const double smear_param) noexcept
            {
                Integrators::GradientFlow::Euler Euler_Integrator(Forcefield);
                GradientFlowKernel               StoutSmear(Gluon, Gluon_smeared, Euler_Integrator, GaugeAction::WilsonAction, smear_param);
                StoutSmear(n_smear_meta);
                return TopChargeClover(Gluon_smeared);
            }
        public:
            explicit MetadynamicsTemperingKernel(GaugeField& U_in, GaugeField& U_temper_in, GaugeField& U_copy1_in, GaugeField& U_copy2_in, MetaBiasPotential& Metapotential_in, prngT& prng_in, double rho_stout_cv_in) noexcept :
            U(U_in), U_temper(U_temper_in), U_copy1(U_copy1_in), U_copy2(U_copy2_in), Metapotential(Metapotential_in), prng(prng_in), rho_stout_cv(rho_stout_cv_in)
            {}

            bool operator()() noexcept
            {
                // TODO: This relies on the assumption that the metapotential always holds the up-to-date CV, might want to change that?
                double CV_old {Metapotential.ReturnCV_current()};
                double CV_new {MetaCharge(U, U_copy1, U_copy2, n_smear_meta, rho_stout_cv)};
                DeltaVTempering = Metapotential.ReturnPotential(CV_new) - Metapotential.ReturnPotential(CV_old);
                double p      {std::exp(-DeltaVTempering)};
                double q      {prng.UniformReal()};
                if (q <= p)
                {
                    // The swap function is cheap since it only swaps the pointers of the underlying raw gaugefields
                    // TODO: Need to change/replace when adding MPI support
                    Swap(U, U_temper);
                    Metapotential.SetCV_current(CV_new);
                    acceptance_count_tempering += 1;
                    return true;
                }
                else
                {
                    return false;
                }
            }
    };
} // namespace GaugeUpdates

#endif // LETTUCE_TEMPERING_HPP
