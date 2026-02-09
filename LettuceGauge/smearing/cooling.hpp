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
//| This file provides a functor implementing a pseudo heat bath based cooling      |
//| algorithm for SU(3) gauge theory.                                               |
//| For more details regarding the pseudo heat bath update and the decomposition    |
//| into SU(2) subgroups check the heat bath and overrelaxation files.              |
//| While cooling is not as theoretically sound as the gradient flow, it leads to   |
//| equivalent results in practice at a faster pace (one cooling sweep reduces the  |
//| action more than a gradient flow step or a smearing step.                       |
//| For more details on the relative performances of smoothing methods see the      |
//| comparisons in arXiv:1401.2441, arXiv:1509.04259, and arXiv:1708.00696.         |
//+---------------------------------------------------------------------------------+

template<typename ActionT>
struct CoolingKernel
{
    private:
        GaugeField& U;
        // TODO: Check if the stencil_radius of the Action is larger than 1 to prevent incorrect masking/parallelization
        ActionT&    Action;
        // Cooling for SU(2) (this function just normalizes the staple)
        template<typename floatT>
        SU2_comp<floatT> CoolingSU2(const SU2_comp<floatT>& A) const noexcept
        {
            floatT a_norm {static_cast<floatT>(1.0) / A.det_sqrt()};
            return a_norm * A.adjoint();
        }
    public:
        explicit CoolingKernel(GaugeField& U_in, ActionT& Action_in) noexcept :
        U(U_in), Action(Action_in)
        {}

        // TODO: Should this be rewritten to be applicable to a general GaugeField?
        //       Would not work with the current implementation, since the reference can't be rebound
        void operator()(const link_coord& current_link) const noexcept
        {
            // Matrix_3x3 W;
            SU2_comp<floatT> subblock;
            // Get the staple
            Matrix_3x3 st_adj {(Action.Staple(U, current_link)).adjoint()};
            //-----
            // Cool (0, 1) subgroup
            subblock        = Extract01<floatT>(U(current_link) * st_adj);
            U(current_link) = Embed01(CoolingSU2(subblock)) * U(current_link);
            //-----
            // Cool (0, 2) subgroup
            subblock        = Extract02<floatT>(U(current_link) * st_adj);
            U(current_link) = Embed02(CoolingSU2(subblock)) * U(current_link);
            //-----
            // Cool (1, 2) subgroup
            subblock        = Extract12<floatT>(U(current_link) * st_adj);
            U(current_link) = Embed12(CoolingSU2(subblock)) * U(current_link);
            //-----
            SU3::Projection::GramSchmidt(U(current_link));
        }
};

#endif // LETTUCE_COOLING_HPP
