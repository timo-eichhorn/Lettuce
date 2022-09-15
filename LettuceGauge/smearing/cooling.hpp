#ifndef LETTUCE_COOLING_HPP
#define LETTUCE_COOLING_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../actions/gauge/wilson_action.hpp"
#include "../math/su2.hpp"
#include "../math/su3.hpp"
//-----
#include <Eigen/Dense>
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
// ...
//----------------------------------------
// Standard C headers
#include <cmath>

//+---------------------------------------------------------------------------------+
//| This file provides a functor implementing a pseudo-heatbath based cooling       |
//| algorithm for SU(3) gauge theory.                                               |
//| For more details regarding the pseudo-heatbath update and the decomposition into|
//| SU(2) subgroups check the heatbath and overrelaxation files.                    |
//| While cooling is not as theoretically sound as the gradient flow, it leads to   |
//| equivalent results in practice at a faster pace (one cooling sweep reduces the  |
//| action more than a gradient flow step or a smearing step.                       |
//| For more details on the relative performances of smoothing methods see the      |
//| comparisons in arXiv:1401.2441, arXiv:1509.04259, and arXiv:1708.00696.         |
//+---------------------------------------------------------------------------------+

// template<typename floatT>
struct CoolingKernel
{
    private:
        // TODO: Change parameter to be action, so we can cool with respect to different actions
        GaugeField& Gluon;
        // Cooling for SU(2) (this function just normalizes the staple)
        template<typename floatT>
        SU2_comp<floatT> CoolingSU2(const SU2_comp<floatT>& A) const noexcept
        {
            floatT a_norm {static_cast<floatT>(1.0) / A.det_sqrt()};
            return a_norm * A.adjoint();
        }
    public:
        explicit CoolingKernel(GaugeField& Gluon_in) noexcept :
        Gluon(Gluon_in)
        {}

        // TODO: Should this be rewritten to be applicable to a general GaugeField?
        //       Would not work with the current implementation, since the reference can't be rebound
        void operator()(const link_coord& current_link) const noexcept
        {
            Matrix_3x3 W;
            SU2_comp<floatT> subblock;
            // Get the staple
            Matrix_3x3 st_adj {WilsonAction::Staple(Gluon, current_link).adjoint()};
            //-----
            // Cool (0, 1) subgroup
            subblock = Extract01<floatT>(Gluon(current_link) * st_adj);
            Gluon(current_link) = Embed01(CoolingSU2(subblock)) * Gluon(current_link);
            //-----
            // Cool (0, 2) subgroup
            subblock = Extract02<floatT>(Gluon(current_link) * st_adj);
            Gluon(current_link) = Embed02(CoolingSU2(subblock)) * Gluon(current_link);
            //-----
            // Cool (1, 2) subgroup
            subblock = Extract12<floatT>(Gluon(current_link) * st_adj);
            Gluon(current_link) = Embed12(CoolingSU2(subblock)) * Gluon(current_link);
            //-----
            SU3::Projection::GramSchmidt(Gluon(current_link));
        }
};

#endif // LETTUCE_COOLING_HPP