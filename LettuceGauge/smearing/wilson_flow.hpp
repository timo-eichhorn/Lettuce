#ifndef LETTUCE_WILSON_FLOW_HPP
#define LETTUCE_WILSON_FLOW_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
// ...
//----------------------------------------
// Standard C headers
// ...

// template<typename GaugeActionT>
// struct WilsonFlowKernel
// {
//     private:
//         // TODO: Possible parameters are epsilon, tau, GaugeAction, ExpFunction...
//     public:
//         WilsonFlowKernel() noexcept :
//         ()
//         {}
//         void operator()(GaugeField& Gluon, const link_coord& current_link) const noexcept
//         {
//             Matrix_3x3 st {WilsonAction::Staple(Gluon, current_link)};
//             Matrix_3x3 A  {st * Gluon(current_link).adjoint()};
//             Matrix_3x3 B  {A - A.adjoint()};
//             Matrix_3x3 C  {static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity()};
//             // Cayley-Hamilton exponential
//             Gluon(current_link) = SU3::exp(-i<floatT> * epsilon * C) * Gluon(current_link);
//             // Eigen exponential (Scaling and squaring)
//             // Gluon(current_link) = (epsilon * C).exp() * Gluon(current_link);
//             // Projection to SU(3) (necessary?)
//             SU3::Projection::GramSchmidt(Gluon(current_link));
//         }
// };

#endif // LETTUCE_WILSON_FLOW_HPP
