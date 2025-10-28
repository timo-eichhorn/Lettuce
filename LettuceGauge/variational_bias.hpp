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

// In theory we could allow an arbitrary functional basis, but restrict to following basis for now:
// alpha_1 * Q^2 + alpha_2 * sin^2(const * Q)
struct SimpleBasis
{
    // TODO: Replace with vector when allowing for general number of parameters
    using ParametersT = std::array<double, 2>;
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

    [[nodiscard]]
    double f0(const double CV) const noexcept
    {
        return CV * CV;
    }

    [[nodiscard]]
    double f1(const double CV) const noexcept
    {
        const double s = std::sin(Constant() * CV);
        return s * s;
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

struct UniformTargetDistribution
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
        return "UniformTargetDistribution";
    }
};

template<typename BasisT, typename TargetDistT = UniformTargetDistribution>
struct VariationalBiasPotential
{
public:
    using ParametersT = BasisT::ParametersT;
private:
    BasisT              functional_basis;
    TargetDistT         target_distribution;
    ParametersT         averaged_parameters;
    std::vector<double> batch;
    double              CV_current;
    double              CV_min;
    double              CV_max;
    double              gradient_descent_stepsize;
    int                 batch_size;
    std::uint64_t       updates_count {0};

    struct MomentsT
    {
        ParametersT mean;
        std::array<std::array<double, 2>, 2> covariance;
    };

    // For the uniform target distribution, we can precompute the expectation values analytically
    [[nodiscard]]
    ParametersT ComputeUniformTargetDistExpectationValues() const noexcept
    {
        const double c = BasisT::Constant();
        const double expectation_Q2   = (CV_max * CV_max + CV_min * CV_min + CV_max * CV_min)/ 3.0;

        const double expectation_sin2 = 0.5 - (std::sin(2.0 * c * CV_max) - std::sin(2.0 * c * CV_min)) / (4.0 * c * (CV_max - CV_min));

        return ParametersT{expectation_Q2, expectation_sin2};
    }

    // [[nodiscard]]
    // ParametersT ComputeBatchExpectationValues() const noexcept
    // {
    //     ParametersT expectation_values{0.0, 0.0};
    //     if (batch.empty())
    //     {
    //         return expectation_values;
    //     }

    //     for (double CV : batch)
    //     {
    //         expectation_values[0] += functional_basis.f0(CV);
    //         expectation_values[1] += functional_basis.f1(CV);
    //     }
    //     const double inverse_N = 1.0 / static_cast<double>(batch.size());
    //     expectation_values[0] *= inverse_N;
    //     expectation_values[1] *= inverse_N;
    //     return expectation_values;
    // }

    [[nodiscard]]
    MomentsT ComputeBatchMoments() const noexcept
    {
        MomentsT moments{};

        std::array<double, 2>                mean{0.0, 0.0};
        std::array<std::array<double, 2>, 2> cov_matrix{{{0.0, 0.0}, {0.0, 0.0}}};

        std::size_t  N = 0;
        for (double CV : batch)
        {
            N += 1;
            const double f0        = functional_basis.f0(CV);
            const double f1        = functional_basis.f1(CV);

            const std::array<double, 2> x{f0, f1};

            std::array<double, 2> delta{x[0] - mean[0], x[1] - mean[1]};

            const double inverse_N = 1.0 / static_cast<double>(N);
            mean[0]               += delta[0] * inverse_N;
            mean[1]               += delta[1] * inverse_N;

            std::array<double, 2> delta2{x[0] - mean[0], x[1] - mean[1]};

            cov_matrix[0][0] += delta[0] * delta2[0];
            cov_matrix[0][1] += delta[0] * delta2[1];
            cov_matrix[1][0] += delta[1] * delta2[0];
            cov_matrix[1][1] += delta[1] * delta2[1];
        }

        if (N == 0)
        {
            moments.mean        = ParametersT{0.0, 0.0};
            moments.covariance  = {{{0.0, 0.0}, {0.0, 0.0}}};
            return moments;
        }

        const double inverse_N = 1.0 / static_cast<double>(N);

        cov_matrix[0][0] *= inverse_N;
        cov_matrix[0][1] *= inverse_N;
        cov_matrix[1][0] *= inverse_N;
        cov_matrix[1][1] *= inverse_N;

        moments.mean       = mean;
        moments.covariance = cov_matrix;
        return moments;
    }

    void UpdateAveragedParameters() noexcept
    {
        for (std::size_t i = 0u; i < averaged_parameters.size(); ++i)
        {
            averaged_parameters[i] = (averaged_parameters[i] * updates_count + functional_basis.parameters[i]) / static_cast<double>(updates_count + 1);
        }
        updates_count += 1;
    }

    void MaybeUpdate() noexcept
    {
        if (static_cast<int>(batch.size()) < batch_size)
        {
            return;
        }
        // const ParametersT batch_expectation_values  = ComputeBatchExpectationValues();
        const MomentsT    batch_moments             = ComputeBatchMoments();
        const ParametersT target_expectation_values = ComputeUniformTargetDistExpectationValues();
        const ParametersT gradient                  = ParametersT{-batch_moments.mean[0] + target_expectation_values[0],
                                                                  -batch_moments.mean[1] + target_expectation_values[1]};

        const double diff_0 = functional_basis.parameters[0] - averaged_parameters[0];
        const double diff_1 = functional_basis.parameters[1] - averaged_parameters[1];

        // Update with averaged stochastic gradient descent algorithm, using the full Hessian in contrast to the original VES paper (where only the diagonals are used)
        functional_basis.parameters[0] -= gradient_descent_stepsize * (gradient[0] + batch_moments.covariance[0][0] * diff_0 + batch_moments.covariance[0][1] * diff_1);
        functional_basis.parameters[1] -= gradient_descent_stepsize * (gradient[1] + batch_moments.covariance[1][0] * diff_0 + batch_moments.covariance[1][1] * diff_1);

        UpdateAveragedParameters();

        batch.clear();
    }
public:

    VariationalBiasPotential(const BasisT& functional_basis_in, const double CV_min_in, const double CV_max_in, const double gradient_descent_stepsize_in, const int batch_size_in) :
    functional_basis(functional_basis_in),
    // averaged_parameters(BasisT::ParametersT.),
    CV_min(CV_min_in),
    CV_max(CV_max_in),
    gradient_descent_stepsize(gradient_descent_stepsize_in),
    batch_size(batch_size_in)
    {
        // averaged_parameters.assign(static_cast<std::size_t>(batch_size), 0.0);
        averaged_parameters = functional_basis.parameters;
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
        batch.push_back(CV);
        MaybeUpdate();

    }

    void UpdatePotential(const std::vector<double>& CV_vec) noexcept
    {
        for (double CV : CV_vec)
        {
            batch.push_back(CV);
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
        // TODO: Replace with basis using averaged parameter values?
        SimpleBasis tmp_basis = functional_basis;
        tmp_basis.parameters  = averaged_parameters;
        return tmp_basis.Evaluate(CV);
        // return functional_basis.Evaluate(CV);
    }

    [[nodiscard]]
    double ReturnDerivative(const double CV) const noexcept
    {
        // TODO: Replace with basis using averaged parameter values?
        SimpleBasis tmp_basis = functional_basis;
        tmp_basis.parameters  = averaged_parameters;
        return tmp_basis.Derivative(CV);
        // return functional_basis.Derivative(CV);
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

    void SaveParameters(const std::string& filename, const bool overwrite = false) const
    {
        std::ofstream ofs;
        if (overwrite)
        {
            ofs.open(filename, std::fstream::out | std::fstream::trunc);
        }
        else
        {
            ofs.open(filename, std::fstream::out | std::fstream::app);
        }

        ofs /*<< "START_VES_PARAMS\n"*/
            << "VES potential"              << "\n"
            << "basis_name:"                << BasisT::GetName()              << "\n"
            << "target_name:"               << TargetDistT::GetName()         << "\n"
            << "CV_min:"                    << CV_min                         << "\n"
            << "CV_max:"                    << CV_max                         << "\n"
            << "gradient_descent_stepsize:" << gradient_descent_stepsize      << "\n"
            << "batch_size:"                << batch_size                     << "\n"
            << "alpha_1:"                   << functional_basis.parameters[0] << "\n"
            << "alpha_2:"                   << functional_basis.parameters[1] << "\n"
            << "alpha_1_avg:"               << averaged_parameters[0]         << "\n"
            << "alpha_2_avg:"               << averaged_parameters[1]         << "\n"
            << "END_VES_PARAMS"                                               << "\n";
    }

    void SavePotential(const std::string& filename, const int grid_points = 801, const bool overwrite = false) const
    {
        std::ofstream ofs;
        if (overwrite)
        {
            ofs.open(filename, std::fstream::out | std::fstream::trunc);
        }
        else
        {
            ofs.open(filename, std::fstream::out | std::fstream::app);
        }

        ofs << "alpha_1:"                   << functional_basis.parameters[0] << "\n"
            << "alpha_2:"                   << functional_basis.parameters[1] << "\n"
            << "alpha_1_avg:"               << averaged_parameters[0]         << "\n"
            << "alpha_2_avg:"               << averaged_parameters[1]         << "\n";
        const double grid_distance = (CV_max - CV_min) / static_cast<double>(grid_points - 1);
        for (int i = 0; i < grid_points - 1; ++i)
        {
            ofs << ReturnPotential(CV_min + grid_distance * i) << ",";
        }
        // ofs << ReturnPotential(CV_min + grid_distance * (grid_points - 1)) << "\n";
        ofs << ReturnPotential(CV_max) << "\n";
    }

    // bool LoadPotential(const std::string& filename) const
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
