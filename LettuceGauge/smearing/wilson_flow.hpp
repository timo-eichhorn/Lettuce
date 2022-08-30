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
struct WilsonFlowKernel
{
    private:
        // TODO: Possible parameters are epsilon, tau, GaugeAction, ExpFunction...
        GaugeField& Gluon;
        floatT epsilon;
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

#endif // LETTUCE_WILSON_FLOW_HPP
