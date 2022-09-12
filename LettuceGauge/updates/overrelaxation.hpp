#ifndef LETTUCE_OVERRELAXATION_HPP
#define LETTUCE_OVERRELAXATION_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../actions/gauge/wilson_action.hpp"
#include "../math/su2.hpp"
#include "../math/su3.hpp"
//-----
#include <Eigen/Dense>
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <random>
//----------------------------------------
// Standard C headers
#include <cmath>

//+---------------------------------------------------------------------------------+
//| This file provides a functor implementing a direct overrelaxation update for    |
//| SU(3) gauge theory, similar to the one described in hep-lat/0409141. The action |
//| is not exactly preserved, so an additional accept-reject step is required       |
//| (with acceptance rates around 0.99). In contrast to the paper above, the        |
//| projection is based on Kenney-Laub iterates (see for example arXiv:1701.00726). |
//| Additionally, another functor implementing a SU(3) overrelaxation based on a    |
//| Cabibbo-Marinari decomposition into SU(2) subgroups is also provided.           |
//+---------------------------------------------------------------------------------+

template<typename floatT>
struct OverrelaxationDirectKernel
{
    private:
        // TODO: Add action as parameter, so we can update with respect to different actions
        GaugeField&                             Gluon;
        std::uniform_real_distribution<floatT>& distribution_prob;
    public:
        explicit OverrelaxationDirectKernel(GaugeField& Gluon_in, std::uniform_real_distribution<floatT>& distribution_prob_in) noexcept :
        Gluon(Gluon_in), distribution_prob(distribution_prob_in)
        {}

        bool operator()(const link_coord& current_link) const noexcept
        {
            Matrix_3x3 st        {WilsonAction::Staple(Gluon, current_link)};
            // Use normalized staple and project onto group via Kenney-Laub projection (using a Gram-Schmidt projection will lead to worse accceptance rates in the end)
            Matrix_SU3 or_matrix {static_cast<floatT>(1.0/6.0) * st};
            SU3::Projection::KenneyLaub(or_matrix);
            Matrix_SU3 old_link  {Gluon(current_link)};
            Matrix_SU3 new_link  {or_matrix * old_link.adjoint() * or_matrix};
            // Calculate action difference
            double     S_old     {WilsonAction::Local(old_link, st)};
            double     S_new     {WilsonAction::Local(new_link, st)};
            double     p         {std::exp(-S_new + S_old)};
            #if defined(_OPENMP)
            double     q         {distribution_prob(prng_vector[omp_get_thread_num()])};
            #else
            double     q         {distribution_prob(generator_rand)};
            #endif
            // The direct overrelaxation algorithm is not exact in the sense that it does not preserve the action, so we need an accept-reject step
            if (q <= p)
            {
                Gluon(current_link) = new_link;
                return 1;
            }
            return 0;
        }
};

// template<typename floatT>
struct OverrelaxationSubgroupKernel
{
    private:
        GaugeField& Gluon;
        // Overrelaxation update for SU(2)
        template<typename floatT>
        SU2_comp<floatT> OverrelaxationSU2(const SU2_comp<floatT>& A) const noexcept
        {
            floatT   a_norm {static_cast<floatT>(1.0) / A.det_sqrt()};
            SU2_comp V      {a_norm * A};
            return (V * V).adjoint();
            // TODO: Replace with version below? They definitely lead to different results when combined with heatbath, but Delta S seems to be similar for both versions
            //       Should probably test the energy violation in more detail and choose the version with smaller <Delta S>
            // floatT a_norm {static_cast<floatT>(1.0) / A.det()};
            // return a_norm * (A * A).adjoint();
        }
    public:
        explicit OverrelaxationSubgroupKernel(GaugeField& Gluon_in) noexcept :
        Gluon(Gluon_in)
        {}

        void operator()(const link_coord& current_link) const noexcept
        {
            SU2_comp<floatT> subblock;
            // Note: Our staple definition corresponds to the daggered staple in Gattringer & Lang, therefore use adjoint
            Matrix_3x3       st_adj {(WilsonAction::Staple(Gluon, current_link)).adjoint()};
            //-----
            // Update (0, 1) subgroup
            // std::cout << "Action before: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
            subblock            = Extract01<floatT>(Gluon(current_link) * st_adj);
            Gluon(current_link) = Embed01(OverrelaxationSU2(subblock)) * Gluon(current_link);
            // std::cout << "Action after: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
            //-----
            // Update (0, 2) subgroup
            // std::cout << "Action before: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
            subblock            = Extract02<floatT>(Gluon(current_link) * st_adj);
            Gluon(current_link) = Embed02(OverrelaxationSU2(subblock)) * Gluon(current_link);
            // std::cout << "Action after: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
            //-----
            // Update (1, 2) subgroup
            // std::cout << "Action before: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
            subblock            = Extract12<floatT>(Gluon(current_link) * st_adj);
            Gluon(current_link) = Embed12(OverrelaxationSU2(subblock)) * Gluon(current_link);
            // std::cout << "Action after: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
            //-----
            SU3::Projection::GramSchmidt(Gluon(current_link));
        }
};

#endif // LETTUCE_OVERRELAXATION_HPP
