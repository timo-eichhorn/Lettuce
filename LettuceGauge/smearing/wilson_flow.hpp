#ifndef LETTUCE_WILSON_FLOW_HPP
#define LETTUCE_WILSON_FLOW_HPP

// Non-standard library headers
#include "../actions/gauge/wilson_action.hpp"
#include "../math/su3.hpp"
#include "../math/su3_exp.hpp"
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
// ...

//+---------------------------------------------------------------------------------+
//| This file provides a functor implementing the gradient flow for SU(3) gauge     |
//| theory integrated with the explicit (first order) Euler integrator.             |
//| For more details see arXiv:0907.5491 and arXiv:1006.4518.                       |
//| TODO: Implement higher order Runge-Kutta integrator and adaptive stepsize.      |
//|       For adaptive stepsize see arXiv:1301.4388.                                |
//+---------------------------------------------------------------------------------+

// template<typename GaugeActionT>
struct WilsonFlowKernel
{
    private:
        // TODO: Possible parameters are epsilon, tau, GaugeAction, Integrator, ExpFunction...
        GaugeField& Gluon;
        floatT      epsilon;
    public:
        explicit WilsonFlowKernel(GaugeField& Gluon_in, const floatT epsilon_in) noexcept :
        Gluon(Gluon_in), epsilon(epsilon_in)
        {}

        // First order explicit Euler integrator (basically equivalent to stout smearing other than the fact that flowed links are immediately used to update other links)
        void operator()(const link_coord& current_link) const noexcept
        {
            // Generally this can be the staple of any action
            // What we want in the end is the algebra-valued derivative of the gauge action, i.e.:
            // T^{a} \partial^{a}_{x, \mu} S_{g}
            // This corresponds to the term C below, i.e., the traceless antihermitian projector applied to the staple
            Matrix_3x3 st {WilsonAction::Staple(Gluon, current_link)};
            Matrix_3x3 A  {st * Gluon(current_link).adjoint()};
            Matrix_3x3 B  {A - A.adjoint()};
            Matrix_3x3 C  {static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity()};
            // Cayley-Hamilton exponential
            Gluon(current_link) = SU3::exp(-i<floatT> * epsilon * C) * Gluon(current_link);
            // Eigen exponential (Scaling and squaring)
            // Gluon(current_link) = (epsilon * C).exp() * Gluon(current_link);
            // Projection to SU(3) (necessary?)
            SU3::Projection::GramSchmidt(Gluon(current_link));
        }

        void SetEpsilon(const floatT epsilon_in) noexcept
        {
            epsilon = epsilon_in;
        }

        floatT GetEpsilon() const noexcept
        {
            return epsilon;
        }
};


// TODO: This is still work in progress
// template<typename floatT>
// template<typename FuncT>
// struct GlobalWilsonFlowKernel
// {
//     private:
//         // TODO: Possible parameters are epsilon, tau, GaugeAction, Integrator, ExpFunction...
//         GaugeField& Gluon;
//         FuncT&      Integrator;
//         floatT      epsilon;

//         // TODO: For higher order integrators, we want to save C since it is used in the following integration steps
//         // void EvolveLink(link_coord& current_link) const noexcept
//         // {
//         //     Matrix_3x3 st {WilsonAction::Staple(Gluon, current_link)};
//         //     Matrix_3x3 A  {st * Gluon(current_link).adjoint()};
//         //     Matrix_3x3 B  {A - A.adjoint()};
//         //     Matrix_3x3 C  {static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity()};
//         //     // Cayley-Hamilton exponential
//         //     Gluon(current_link) = SU3::exp(-i<floatT> * epsilon * C) * Gluon(current_link);
//         //     // Eigen exponential (Scaling and squaring)
//         //     // Gluon(current_link) = (epsilon * C).exp() * Gluon(current_link);
//         //     // Projection to SU(3) (necessary?)
//         //     SU3::Projection::GramSchmidt(Gluon(current_link));
//         // }

//         void CalculateZ(const GaugeField& Gluon, GaugeField& Z) const noexcept
//         {
//             // TODO: Boilerplate loops or rewrite using Iterator?
//             for (int mu = 0; mu < 4; ++mu)
//             for (int eo = 0; eo < 2; ++eo)
//             {
//                 #pragma omp parallel for
//                 for (int t = 0; t < Nt; ++t)
//                 for (int x = 0; x < Nx; ++x)
//                 for (int y = 0; y < Ny; ++y)
//                 {
//                     int offset {((t + x + y) & 1) ^ eo};
//                     for (int z = offset; z < Nz; z+=2)
//                     {
//                         Matrix_3x3 st {WilsonAction::Staple(Gluon, current_link)};
//                         Matrix_3x3 A  {st * Gluon(current_link).adjoint()};
//                         Z(current_link) = SU3::Projection::Algebra(A);
//                     }
//                 }
//             }
//         }

//         void EvolveLink(GaugeField& Gluon, const GaugeField& Z) const noexcept
//         {
//             // TODO: Boilerplate loops or rewrite using Iterator?
//             for (int mu = 0; mu < 4; ++mu)
//             for (int eo = 0; eo < 2; ++eo)
//             {
//                 #pragma omp parallel for
//                 for (int t = 0; t < Nt; ++t)
//                 for (int x = 0; x < Nx; ++x)
//                 for (int y = 0; y < Ny; ++y)
//                 {
//                     int offset {((t + x + y) & 1) ^ eo};
//                     for (int z = offset; z < Nz; z+=2)
//                     {
//                         // Cayley-Hamilton exponential
//                         Gluon(current_link) = SU3::exp(-i<floatT> * epsilon * Z) * Gluon(current_link);
//                         // Eigen exponential (Scaling and squaring)
//                         // Gluon(current_link) = (epsilon * Z).exp() * Gluon(current_link);
//                         // Projection to SU(3) (necessary?)
//                         SU3::Projection::GramSchmidt(Gluon(current_link));
//                     }
//                 }
//             }
//         }

//         void RK2(GaugeField& Gluon) const noexcept
//         {
//             //...
//             // W_0             = V_t
//             // W_1             = exp(1/4 * Z_0)      * W_0
//             // V_{t + epsilon} = exp(2 * Z_1 - Z_0)  * W_0
//             // Z_i = epsilon * Z(W_i)
//         }

//         void RK3(GaugeField& Gluon) const noexcept
//         {
//             // W_0             = V_t
//             // W_1             = exp(1/4 * Z_0)                           * W_0
//             // W_2             = exp(8/9 * Z_1 - 17/36 * Z_0)             * W_1
//             // V_{t + epsilon} = exp(3/4 * Z_2 - 8/9 * Z_1 + 17/36 * Z_0) * W_2
//             // Z_i = epsilon * Z(W_i)
//             static GaugeField Z_temp;
//             CalculateZ(Gluon, Z_temp);
//         }
//     public:
//         explicit GlobalWilsonFlowKernel(GaugeField& Gluon_in, FuncT& Integrator_in, const floatT epsilon_in) noexcept :
//         Gluon(Gluon_in), Integrator(Integrator_in), epsilon(epsilon_in)
//         {}

//         void operator()() const noexcept
//         {
//             // TODO: Add more arguments to EvolveLink (like a reference to an array holding the force, and then bind those parameters using a lambda?)
//             Iterator::Checkerboard(Gluon, EvolveLink);
//         }

//         void SetEpsilon(const floatT epsilon_in) noexcept
//         {
//             epsilon = epsilon_in;
//         }

//         floatT GetEpsilon() const noexcept
//         {
//             return epsilon;
//         }
// };

#endif // LETTUCE_WILSON_FLOW_HPP
