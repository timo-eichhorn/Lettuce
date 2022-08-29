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

// TODO: Currently this seems to lead to some weird results. The action seems to monotonically decrease, but the topological charge (and other observables)
//       show some significant deviations when compared to stout smearing. Additionally, while we would generally expect cooling to decrease the action faster
//       than smearing, this only seems to be the case initially; after some amount of smoothing smearing seems to be more efficient, while cooling seemingly
//       slows down.
// The operations below don't really make much sense... Iterating over the three subgroups is superfluous, since only the last iteration is actually relevant,
// since in the current form the new link doesn't depend on the old one in any way. Might need to multiply with the old link?

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

        void operator()(const link_coord& current_link) const noexcept
        // void operator()(GaugeField& Gluon, const link_coord& current_link) const noexcept
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
            // Project link to SU(3)
            SU3::Projection::GramSchmidt(Gluon(current_link));
        }
};

#endif // LETTUCE_COOLING_HPP
