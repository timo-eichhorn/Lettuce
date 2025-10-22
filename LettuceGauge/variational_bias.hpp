#ifndef LETTUCE_VARIATIONAL_BIAS_HPP
#define LETTUCE_VARIATIONAL_BIAS_HPP

// Non-standard library headers
#include "defines.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <string>
#include <vector>
//----------------------------------------
// Standard C headers
#include <cmath>

// Need to store previous measurements, along with actions to allow for weighting the measurements correctly?

// TODO: In theory we could allow an arbitrary functional basis, but restrict to Fourier basis for now?
struct SimpleBasis
{
    std::vector<double> parameters;
    const double renormalization_factor {0.84};

    static std::string GetName()
    {
        return "Quadratic_Sin^2_" + std::to_string(renormalization_factor);
    }

    // Only 1D CV space
    [[nodiscard]]
    double Evaluate(const double CV) const
    {
        const double sin_term = std::sin(pi<double> / renormalization_factor * CV);
        return parameters[0] * CV * CV + parameters[1] * sin_term * sin_term;
    }
}

struct VariationalBiasPotential
{
private:
    // TODO: Need a way to specify the target distribution

    std::vector<double> averaged_parameter_set; // TODO: Probably keep this here
    double gradient_descent_stepsize;
    int    batch_size;
public:

    // TODO: Can probably be done in a better way than returning a vector?
    std::vector<double> ComputeGradient() noexcept
    {
        // - expectation value of derivative sampled with V + expectation value of derivative sampled according to target distribution p
    }

    // TODO: Can probably be done in a better way than returning a vector?
    std::vector<double> ComputeHessianTerm() noexcept
    {
        // Covariance of derivatives + terms that vanish when when the bias is a liner combination of basis functions
        // Never compute full Hessian, directly multiply with difference vector to get a vector again
        Hessian(current_averaged_parameter_set) * (current_parameter_set - current_averaged_parameter_set);
    }

    void UpdatePotential(const double CV) noexcept
    {
        // Use an averaged stochastic gradient descent algorithm
        // TODO: Can't update in-place? Or have to precompute gradient and hessian
        auto gradient;
        auto hessian;
        for (auto parameter : parameters)
        {
            parameter -= gradient_descent_stepsize * (ComputeGradient(current_averaged_parameter_set) + ComputeHessianTerm(current_averaged_parameter_set));
        }
    }

    void UpdatePotential(const std::vector<double>& CV_vec) noexcept
    {
        //
    }

    // TODO: Is this needed?
    // void UpdatePotentialSymmetric(const double CV) noexcept
    // {
    //     //
    // }

    // void UpdatePotentialSymmetric(const std::vector<double>& CV_vec) noexcept
    // {
    //     //
    // }

    [[nodiscard]]
    double ReturnPotential(const double CV) const noexcept
    {
        //
    }

    [[nodiscard]]
    double ReturnDerivative(const double CV) const noexcept
    {
        //
    }

    void SaveParameters(const std::string& filename, const bool overwrite = false)
    {
        //
    }

    void SavePotential(const std::string& filename, const bool overwrite = false)
    {
        //
    }

    bool LoadPotential(const std::string& filename)
    {
        //
    }
};

#endif // LETTUCE_VARIATIONAL_BIAS_HPP
