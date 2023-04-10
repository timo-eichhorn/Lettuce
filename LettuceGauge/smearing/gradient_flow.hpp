#ifndef LETTUCE_GRADIENT_FLOW_HPP
#define LETTUCE_GRADIENT_FLOW_HPP

// Non-standard library headers
#include "../math/su3.hpp"
#include "../math/su3_exp.hpp"
//-----
#include <Eigen/Dense>
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
// ...
//----------------------------------------
// Standard C headers
// ...

//+---------------------------------------------------------------------------------+
//| This file provides a functor implementing the gradient flow for SU(3) gauge     |
//| theory, supporting generic integrators and actions. In addition, three integra- |
//| tors (Euler, RK2, RK3) are provided.                                            |
//| For more details see arXiv:0907.5491 and arXiv:1006.4518.                       |
//| TODO: Implement adaptive stepsize integrator (see arXiv:1301.4388).             |
//+---------------------------------------------------------------------------------+

namespace Integrators::GradientFlow
{
    // TODO: Implement
    struct Euler
    {
        GaugeField& Force;

        explicit Euler(GaugeField& Force_in) noexcept :
        Force(Force_in)
        {}

        template<typename GradientFlowFunctor>
        void operator()(GradientFlowFunctor& GradientFlow, const int n_step) const noexcept
        {
            // W_0             = V_t
            // V_{t + epsilon} = exp(Z_0)  * W_0
            // Z_i = epsilon * Z(W_i)
            for (int step_count = 0; step_count < n_step; ++step_count)
            {
                GradientFlow.CalculateZ(GradientFlow.U_flowed, Force, GradientFlow.epsilon);
                GradientFlow.UpdateFields(GradientFlow.U_flowed, Force, 1.0);
            }
        }

        static std::string ReturnName()
        {
            return "Euler/StoutSmearing";
        }
    };

    // TODO: This seems to be incorrect
    // struct Midpoint
    // {
    //     template<typename GradientFlowFunctor>
    //     void operator()(GradientFlowFunctor& GradientFlow, const int n_step) const noexcept
    //     {
    //         // W_0             = V_t
    //         // W_1             = exp(1/2 * Z_0)      * W_0
    //         // V_{t + epsilon} = exp(Z_1)            * W_0
    //         // Z_i = epsilon * Z(W_i)
    //         for (int step_count = 0; step_count < n_step; ++step_count)
    //         {
    //             // GradientFlow.CalculateZ(GradientFlow.U_flowed, GradientFlow.Force, GradientFlow.epsilon);
    //             // GradientFlow.UpdateZ(GradientFlow.U_flowed, GradientFlow.Force, -1.0, 2.0 * GradientFlow.epsilon);
    //             // GradientFlow.UpdateFields(GradientFlow.U_flowed, GradientFlow.Force, 1.0);
    //             GradientFlow.CalculateZ(GradientFlow.U_flowed, GradientFlow.Force, GradientFlow.epsilon);
    //             GradientFlow.UpdateFields(GradientFlow.U_flowed, GradientFlow.Force, 0.5);

    //             GradientFlow.CalculateZ(GradientFlow.U_flowed, GradientFlow.Force, GradientFlow.epsilon);
    //             // GradientFlow.UpdateZ(GradientFlow.U_flowed, GradientFlow.Force, 0.0, 1.0 * GradientFlow.epsilon);
    //             GradientFlow.UpdateFields(GradientFlow.U_flowed, GradientFlow.Force, 1.0);
    //         }
    //     }

    //     static std::string ReturnName()
    //     {
    //         return "Midpoint";
    //     }
    // };

    // TODO: For smaller step sizes, this seems to perform better than the Euler integrator, but not clear for larger step sizes (check!)
    struct RK2
    {
        GaugeField& Force;

        explicit RK2(GaugeField& Force_in) noexcept :
        Force(Force_in)
        {}

        template<typename GradientFlowFunctor>
        void operator()(GradientFlowFunctor& GradientFlow, const int n_step) const noexcept
        {
            // W_0             = V_t
            // W_1             = exp(1/2 * Z_0)        * W_0
            // V_{t + epsilon} = exp(Z_1 - 1/2 * Z_0)  * W_0
            // Z_i = epsilon * Z(W_i)
            for (int step_count = 0; step_count < n_step; ++step_count)
            {
                GradientFlow.CalculateZ(GradientFlow.U_flowed, Force, GradientFlow.epsilon);
                GradientFlow.UpdateFields(GradientFlow.U_flowed, Force, 0.5);

                GradientFlow.UpdateZ(GradientFlow.U_flowed, Force, -0.5, GradientFlow.epsilon);
                GradientFlow.UpdateFields(GradientFlow.U_flowed, Force, 1.0);
            }
        }

        static std::string ReturnName()
        {
            return "RK2";
        }
    };

    struct RK3
    {
        GaugeField& Force;

        explicit RK3(GaugeField& Force_in) noexcept :
        Force(Force_in)
        {}

        template<typename GradientFlowFunctor>
        void operator()(GradientFlowFunctor& GradientFlow, const int n_step) const noexcept
        {
            // W_0             = V_t
            // W_1             = exp(1/4 * Z_0)                           * W_0
            // W_2             = exp(8/9 * Z_1 - 17/36 * Z_0)             * W_1
            // V_{t + epsilon} = exp(3/4 * Z_2 - 8/9 * Z_1 + 17/36 * Z_0) * W_2
            // Z_i = epsilon * Z(W_i)
            for (int step_count = 0; step_count < n_step; ++step_count)
            {
                GradientFlow.CalculateZ(GradientFlow.U_flowed, Force, GradientFlow.epsilon);
                GradientFlow.UpdateFields(GradientFlow.U_flowed, Force, 0.25);

                GradientFlow.UpdateZ(GradientFlow.U_flowed, Force, -17.0/36.0, 8.0/9.0 * GradientFlow.epsilon);
                GradientFlow.UpdateFields(GradientFlow.U_flowed, Force, 1.0);

                GradientFlow.UpdateZ(GradientFlow.U_flowed, Force, -1.0, 3.0/4.0 * GradientFlow.epsilon);
                GradientFlow.UpdateFields(GradientFlow.U_flowed, Force, 1.0);
            }
        }

        static std::string ReturnName()
        {
            return "RK3";
        }
    };

    // TODO: Implement here or as separate functor (adaptive integration needs more memory than standard integrators)?
    //       Second order integrator looks something like this:
    //       W_0             = V_t
    //       W_1             = exp(1/4 * Z_0)      * W_0
    //       V_{t + epsilon} = exp(2 * Z_1 - Z_0)  * W_0
    //       Z_i = epsilon * Z(W_i)
    // TODO: It is not yet clear to me how the interface should look like. While the simple fixed stepsize integrators obviously return
    //       fields at evenly spaced flow times, adaptive stepsize integrators do not. Should we pass predetermined flow times at which
    //       to calculate observables?
    // struct RK3_adaptive
    // {
    //     GaugeField& Force;
    //     GaugeField& U_flowed_lower_order;

    //     template<typename GradientFlowFunctor>
    //     void operator()(GradientFlowFunctor& GradientFlow, const int n_step) noexcept
    //     {
    //         for (int step_count = 0; step_count < n_step; ++step_count)
    //         {
    //             //
    //             GradientFlow.CalculateZ(GradientFlow.U_flowed, Force, GradientFlow.epsilon);
    //         }
    //     }

    //     static std::string ReturnName()
    //     {
    //         return "RK3_adaptive";
    //     }
    // };
} // namespace Integrators::GradientFlow

// template<typename floatT>
template<typename IntegratorT, typename ActionT>
struct GradientFlowKernel
{
    private:
        // TODO: Possible parameter ExpFunction?
        const GaugeField&  U_unflowed;
              GaugeField&  U_flowed;
              IntegratorT& Integrator;
              ActionT&     Action;
              floatT       epsilon;

        // The integrator needs to access private member functions
        friend IntegratorT;

        void UpdateFields(GaugeField& U, const GaugeField& Z, const floatT c) const noexcept
        {
            #pragma omp parallel for
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
            for (int mu = 0; mu < 4; ++mu)
            {
                link_coord current_link {t, x, y, z, mu};
                // Cayley-Hamilton exponential
                U(current_link) = SU3::exp(-i<floatT> * c * Z(current_link)) * U(current_link);
                // Eigen exponential (Scaling and squaring)
                // U(current_link) = (c * Z(current_link)).exp() * U(current_link);
                // Projection to SU(3) (necessary?)
                SU3::Projection::GramSchmidt(U(current_link));
            }
        }

        // void AddForces(GaugeField& Z_1, const GaugeField& Z_2, const floatT c_1, const floatT c_2) const noexcept
        // {
        //     #pragma omp parallel for
        //     for (int t = 0; t < Nt; ++t)
        //     for (int x = 0; x < Nx; ++x)
        //     for (int y = 0; y < Ny; ++y)
        //     for (int z = 0; z < Nz; ++z)
        //     for (int mu = 0; mu < 4; ++mu)
        //     {
        //         link_coord current_link {t, x, y, z, mu};
        //         Z_1(current_link) = c_1 * Z_1(current_link) + c_2 * Z_2(current_link);
        //     }
        // }

        void CalculateZ(const GaugeField& U, GaugeField& Z, const floatT epsilon) const noexcept
        {
            #pragma omp parallel for
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
            {
                for (int mu = 0; mu < 4; ++mu)
                {
                    link_coord current_link {t, x, y, z, mu};
                    Matrix_3x3 st {Action.Staple(U, current_link)};
                    Matrix_3x3 A  {st * U(current_link).adjoint()};
                    Z(current_link) = epsilon * SU3::Projection::Algebra(A);
                }
            }
        }

        void UpdateZ(const GaugeField& U, GaugeField& Z, const floatT c_old, const floatT c_new) const noexcept
        {
            #pragma omp parallel for
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
            {
                for (int mu = 0; mu < 4; ++mu)
                {
                    link_coord current_link {t, x, y, z, mu};
                    Matrix_3x3 st {Action.Staple(U, current_link)};
                    Matrix_3x3 A  {st * U(current_link).adjoint()};
                    Z(current_link) = c_old * Z(current_link) + c_new * SU3::Projection::Algebra(A);
                }
            }
        }

    public:
        explicit GradientFlowKernel(const GaugeField& U_unflowed_in, GaugeField& U_flowed_in, IntegratorT& Integrator_in, ActionT& Action_in, const floatT epsilon_in) noexcept :
        U_unflowed(U_unflowed_in), U_flowed(U_flowed_in), Integrator(Integrator_in), Action(Action_in), epsilon(epsilon_in)
        {}

        void operator()(const int n_step) const noexcept
        {
            // TODO: Add more arguments to EvolveLink (like a reference to an array holding the force, and then bind those parameters using a lambda?)
            // Iterator::Checkerboard(U_unflowed, EvolveLink);
            U_flowed = U_unflowed;
            Integrator(*this, n_step);
        }

        void Resume(const int n_step) const noexcept
        {
            // Integrate without copying U_unflowed into U_flowed
            Integrator(*this, n_step);
        }

        void SetEpsilon(const floatT epsilon_in) noexcept
        {
            epsilon = epsilon_in;
        }

        floatT GetEpsilon() const noexcept
        {
            return epsilon;
        }

        std::string ReturnIntegratorName() const
        {
            return Integrator.ReturnName();
        }
};

#endif // LETTUCE_GRADIENT_FLOW_HPP
