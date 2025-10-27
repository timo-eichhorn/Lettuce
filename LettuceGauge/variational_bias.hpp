#ifndef LETTUCE_VARIATIONAL_BIAS_HPP
#define LETTUCE_VARIATIONAL_BIAS_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>
//----------------------------------------
// Standard C headers
#include <cmath>

// Need to store previous measurements, along with actions to allow for weighting the measurements correctly?

// TODO: In theory we could allow an arbitrary functional basis, but restrict to following basis for now:
//       alpha_1 * Q^2 + alpha_2 * sin^2(const * Q)
struct SimpleBasis
{
    using ParametersT = std::array<double, 2>;
    // TODO: Replace with vector when allowing for general number of parameters
    ParametersT             parameters             {0.0, 0.0};
    static constexpr double renormalization_factor {0.84};

    [[nodiscard]]
    static constexpr double Constant() noexcept
    {
        return std::numbers::pi_v<double> / renormalization_factor;
    }

    [[nodiscard]]
    static std::string GetName()
    {
        return "Quadratic_Sin^2_" + std::to_string(renormalization_factor);
    }

    // Only 1D CV space
    [[nodiscard]]
    double Evaluate(const double CV) const noexcept
    {
        const double sin_term = std::sin(Constant() * CV);
        return parameters[0] * CV * CV + parameters[1] * sin_term * sin_term;
    }

    [[nodiscard]]
    double Derivative(const double CV) const noexcept
    {
        // const double arg = Constant() * CV;
        // return 2 * (parameters[0] * CV + parameters[1] * Constant() * std::sin(arg) * std::cos(arg));
        return 2.0 * parameters[0] * CV + parameters[1] * Constant() * std::sin(2.0 * Constant() * CV);
    }
};

struct TargetDistributionUniform
{
    // TODO: Add boundaries CV_min CV_max, outside of which the distribution is zero/falls off exponentially?
    // Only needs to be proportional to the target density, so unnormalized is fine?
    [[nodiscard]]
    double operator()(const double /*CV*/) const noexcept
    {
        return 1.0;
    }

    [[nodiscard]]
    static std::string GetName()
    {
        return "TargetDistributionUniform";
    }
};

template<typename BasisT, typename TargetDistT = TargetDistributionUniform>
struct VariationalBiasPotential
{
public:
    using ParametersT = BasisT::ParametersT;
private:
    BasisT              functional_basis;
    std::vector<double> averaged_parameters;
    std::vector<double> batch;
    double              CV_current;
    double              CV_min;
    double              CV_max;
    double              gradient_descent_stepsize;
    int                 batch_size;

    // double Reweight(const std::vector<std::pair<double, double>>& measurements) const noexcept
    // {
    //     // Pair CV measurements with action measurements
    //     // return
    // }

    void MaybeUpdate() noexcept
    {
        if (static_cast<int>(batch.size()) < batch_size)
        {
            return;
        }
        // TODO: Perform update and clear batch
    }

    // TODO: Can probably be done in a better way than returning a vector?
    ParametersT/*std::vector<double>*/ ComputeGradient() const noexcept
    {
        // - expectation value of derivative sampled with V + expectation value of derivative sampled according to target distribution p
    }

    // TODO: Can probably be done in a better way than returning a vector?
    ParametersT/*std::vector<double>*/ ComputeHessianTerm() const noexcept
    {
        // Covariance of derivatives + terms that vanish when when the bias is a liner combination of basis functions
        // Never compute full Hessian, directly multiply with difference vector to get a vector again
        // The Hessian can be computed exactly for simple target distributions
        Hessian(current_averaged_parameters) * (current_parameters - current_averaged_parameters);
    }
public:

    VariationalBiasPotential(const BasisT& functional_basis_in, const double CV_min_in, const double CV_max_in, const double gradient_descent_stepsize_in, const int batch_size_in) :
    functional_basis(functional_basis_in),
    CV_min(CV_min_in),
    CV_max(CV_max_in),
    gradient_descent_stepsize(gradient_descent_stepsize_in),
    batch_size(batch_size_in)
    {
        averaged_parameters.assign(static_cast<std::size_t>(batch_size), 0.0);
        std::cout << "\nInitialized VariationalBiasPotential with the following parameters:\n"
                  << "  functional_basis_name:     " << BasisT::GetName()         << "\n"
                  << "  target_distribution_name:  " << TargetDistT::GetName()    << "\n"
                  << "  CV_min:                    " << CV_min                    << "\n"
                  << "  CV_max:                    " << CV_max                    << "\n"
                  << "  gradient_descent_stepsize: " << gradient_descent_stepsize << "\n"
                  << "  batch_size:                " << batch_size                << "\n" << std::endl;
    }

    void UpdatePotential(const double CV) noexcept
    {
        // Use an averaged stochastic gradient descent algorithm
        // TODO: Store CV (and action?) value, only update if sufficient samples (batch_size) have been collected
        // TODO: Can't update in-place? Or have to precompute gradient and hessian
        // auto gradient;
        // auto hessian;
        // for (auto parameter : parameters)
        // {
        //     parameter -= gradient_descent_stepsize * (ComputeGradient(current_averaged_parameters) + ComputeHessianTerm(current_averaged_parameters));
        // }
        batch.push_back(CV);
        MaybeUpdate();

    }

    void UpdatePotential(const std::vector<double>& CV_vec) noexcept
    {
        for (double el : CV_vec)
        {
            batch.push_back(el);
        }
        MaybeUpdate();
    }

    // TODO: Is this needed? Symmetry is already built into the functional basis
    void UpdatePotentialSymmetric(const double CV) noexcept
    {
        UpdatePotential(CV);
    }

    void UpdatePotentialSymmetric(const std::vector<double>& CV_vec) noexcept
    {
        UpdatePotential(CV_vec);
    }

    [[nodiscard]]
    double ReturnPotential(const double CV) const noexcept
    {
        return functional_basis.Evaluate(CV);
    }

    [[nodiscard]]
    double ReturnDerivative(const double CV) const noexcept
    {
        return functional_basis.Derivative(CV);
    }

    void SetCV_current(const double CV_in) noexcept
    {
        CV_current = CV_in;
    }

    [[nodiscard]]
    double ReturnCV_current() const noexcept
    {
        return CV_current;
    }

    [[nodiscard]]
    ParametersT ReturnParameters() const noexcept
    {
        return functional_basis.parameters;
    }

    [[nodiscard]]
    ParametersT ReturnAveragedParameters() const noexcept
    {
        return averaged_parameters;
    }

    // void SaveParameters(const std::string& filename, const bool overwrite = false)
    // {
    //     std::ofstream ofs;
    //     if (overwrite)
    //     {
    //         ofs.open(filename, std::fstream::out | std::fstream::trunc);
    //     }
    //     else
    //     {
    //         ofs.open(filename, std::fstream::out | std::fstream::app);
    //     }
    // }

    // void SavePotential(const std::string& filename, const bool overwrite = false)
    // {
    //     std::ofstream ofs;
    //     if (overwrite)
    //     {
    //         ofs.open(filename, std::fstream::out | std::fstream::trunc);
    //     }
    //     else
    //     {
    //         ofs.open(filename, std::fstream::out | std::fstream::app);
    //     }
    // }

    // bool LoadPotential(const std::string& filename)
    // {
    //     if (!std::filesystem::exists(filename))
    //     {
    //         std::cerr << Lettuce::Color::BoldRed << "File " << filename << " not found!" << Lettuce::Color::Reset << "\n";
    //         return false;
    //     }

    //     std::ifstream ifs;
    //     ifs.open(filename, std::fstream::in);
    //     if (!ifs)
    //     {
    //         std::cerr << Lettuce::Color::BoldRed << "Reading potential from file " << filename << " failed!" << Lettuce::Color::Reset << "\n";
    //         return false;
    //     }
    // }
};

#endif // LETTUCE_VARIATIONAL_BIAS_HPP
