#ifndef LETTUCE_TEMPERING_HPP
#define LETTUCE_TEMPERING_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../lattice.hpp"
#include "../metadynamics.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <utility>
#include <complex>
//----------------------------------------
// Standard C headers
#include <cmath>


bool MetadynamicsTempering(GaugeField& Gluon, GaugeField& Gluon1, GaugeField& Gluon2, MetaBiasPotential& Metapotential, GaugeField& GluonMeta, uint_fast64_t& acceptance_count_tempering, const bool metropolis_test, std::uniform_real_distribution<floatT>& distribution_prob, std::uniform_real_distribution<floatT>& distribution_uniform) noexcept
{
    // Actual update
    double CV_new = Metapotential.CV_current;
    double CV_old = {MetaCharge(Gluon, Gluon1, Gluon2, n_smear_meta, rho_stout)};
    #if defined(_OPENMP)
    double q     {distribution_prob(prng_vector[omp_get_thread_num()])};
    #else
    double q     {distribution_prob(generator_rand)};
    #endif

    DeltaV {Metapotential.ReturnPotential(CV_new) - Metapotential.ReturnPotential(CV_old)};
    double p {std::exp(-DeltaV)};
    if (metropolis_test)
    {
        if (q <= p)
        {
            acceptance_count_tempering += 1;
            std::swap(Gluon,GluonMeta);
            if constexpr(metapotential_updated)
            {
                Metapotential.UpdatePotential(CV_old);
            }
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        std::swap(Gluon,GluonMeta);
        if constexpr(metapotential_updated)
        {
            Metapotential.UpdatePotential(CV_old);
        }
        datalog << "DeltaV: " << DeltaV << std::endl;
        return true;
    }
}

#endif // LETTUCE_TEMPERING_HPP
