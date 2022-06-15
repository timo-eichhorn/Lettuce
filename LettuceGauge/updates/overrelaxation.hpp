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

struct OverrelaxationDirectKernel
{
    private:
        GaugeField& Gluon;
        std::uniform_real_distribution<floatT>& distribution_prob;
    public:
        explicit OverrelaxationDirectKernel(GaugeField& Gluon_in, std::uniform_real_distribution<floatT>& distribution_prob_in) noexcept :
        Gluon(Gluon_in), distribution_prob(distribution_prob_in)
        {}

        bool operator()(const link_coord& current_link) const noexcept
        {
            Matrix_3x3 st {WilsonAction::Staple(Gluon, current_link.t, current_link.x, current_link.y, current_link.z, current_link.mu)};
            // Use normalized staple
            Matrix_SU3 or_matrix {static_cast<floatT>(1.0/6.0) * st};
            SU3::Projection::KenneyLaub(or_matrix);
            Matrix_SU3 old_link {Gluon(current_link)};
            Matrix_SU3 new_link {or_matrix * old_link.adjoint() * or_matrix};

            double s      {WilsonAction::Local(old_link, st)};
            double sprime {WilsonAction::Local(new_link, st)};
            double p      {std::exp(-sprime + s)};
            #if defined(_OPENMP)
            double q {distribution_prob(prng_vector[omp_get_thread_num()])};
            #else
            double q {distribution_prob(generator_rand)};
            #endif
            // The direct overrelaxation algorithm is not exact in the sense that it does not preserve the action, so we need an accept-reject step
            if (q <= p)
            {
                Gluon(current_link) = new_link;
                s = sprime;
                return 1;
                // acceptance_count_or += 1;
            }
            return 0;
        }
};

struct OverrelaxationSubgroupKernel
{
    private:
        GaugeField& Gluon;
        // Overrelaxation update for SU(2)
        template<typename floatT>
        SU2_comp<floatT> OverrelaxationSU2(const SU2_comp<floatT>& A) const noexcept
        {
            floatT a_norm {static_cast<floatT>(1.0) / std::sqrt(A.det_sq())};
            SU2_comp V {a_norm * A};
            return (V * V).adjoint();
        }
    public:
        explicit OverrelaxationSubgroupKernel(GaugeField& Gluon_in) noexcept :
        Gluon(Gluon_in)
        {}

        void operator()(const link_coord& current_link) const noexcept
        {
            Matrix_3x3 W;
            SU2_comp<floatT> subblock;
            // Note: Our staple definition corresponds to the daggered staple in Gattringer & Lang, therefore use adjoint
            Matrix_3x3 st_adj {(WilsonAction::Staple(Gluon, current_link.t, current_link.x, current_link.y, current_link.z, current_link.mu)).adjoint()};
            //-----
            // Update (0, 1) subgroup
            // W = Gluon[t][x][y][z][mu] * st_adj;
            // std::cout << "Action before: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
            subblock = Extract01<floatT>(Gluon(current_link) * st_adj);
            Gluon(current_link) = Embed01(OverrelaxationSU2(subblock)) * Gluon(current_link);
            // std::cout << "Action after: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
            //-----
            // Update (0, 2) subgroup
            // W = Gluon[t][x][y][z][mu] * st_adj;
            // std::cout << "Action before: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
            subblock = Extract02<floatT>(Gluon(current_link) * st_adj);
            Gluon(current_link) = Embed02(OverrelaxationSU2(subblock)) * Gluon(current_link);
            // std::cout << "Action after: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
            //-----
            // Update (1, 2) subgroup
            // W = Gluon[t][x][y][z][mu] * st_adj;
            // std::cout << "Action before: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
            subblock = Extract12<floatT>(Gluon(current_link) * st_adj);
            Gluon(current_link) = Embed12(OverrelaxationSU2(subblock)) * Gluon(current_link);
            // std::cout << "Action after: " << SLocal(Gluon[t][x][y][z][mu], st_adj.adjoint()) << endl;
            //-----
            // Project link to SU(3)
            SU3::Projection::GramSchmidt(Gluon(current_link));
        }
};

#endif // LETTUCE_OVERRELAXATION_HPP
