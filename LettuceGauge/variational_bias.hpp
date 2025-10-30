#ifndef LETTUCE_VARIATIONAL_BIAS_HPP
#define LETTUCE_VARIATIONAL_BIAS_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <algorithm>
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
#include <cstdint>

// Need to store previous measurements, along with actions to allow for weighting the measurements correctly?

// In theory we could allow an arbitrary functional basis, but restrict to following basis for now:
// alpha_1 * Q^2 + alpha_2 * sin^2(const * Q)
struct SimpleBasis
{
    // TODO: Replace with vector when allowing for general number of parameters
    static constexpr int    n_parameters = 2;
    using ParametersT = std::array<double, n_parameters>;
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

    [[nodiscard]]
    double f0_deriv(const double CV) const noexcept
    {
        return 2.0 * CV;
    }

    [[nodiscard]]
    double f1_deriv(const double CV) const noexcept
    {
        return Constant() * std::sin(2.0 * Constant() * CV);
    }

    // Only 1D CV space
    [[nodiscard]]
    double Evaluate(const double CV) const noexcept
    {
        return parameters[0] * f0(CV) + parameters[1] * f1(CV);
    }

    [[nodiscard]]
    double Derivative(const double CV) const noexcept
    {
        return parameters[0] * f0_deriv(CV) + parameters[1] * f1_deriv(CV);
    }
};

struct UniformTargetDistribution
{
    // TODO: Add boundaries CV_min CV_max, outside of which the distribution is zero/falls off exponentially?
    // Only needs to be proportional to the target density, so unnormalized is fine
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

// If only the barrier terms are removed, the resulting distribution should be approximately Gaussian
struct GaussianTargetDistribution
{
    double mean               {0.0};
    double standard_deviation {1.0};

    // Only needs to be proportional to the target density, so unnormalized is fine
    [[nodiscard]]
    double operator()(const double CV) const noexcept
    {
        const double arg = (CV - mean) / standard_deviation;
        return std::exp(-0.5 * arg * arg);
    }

    [[nodiscard]]
    static std::string GetName()
    {
        return "GaussianTargetDistribution";
    }
};

template<typename BasisT, typename TargetDistT>
struct VariationalBiasPotential
{
public:
    using ParametersT = BasisT::ParametersT;
    static constexpr int n_parameters = BasisT::n_parameters;
private:
    BasisT              functional_basis;
    TargetDistT         target_distribution;
    ParametersT         averaged_parameters;
    ParametersT         target_distribution_expectation_values;
    double              CV_current;
    double              CV_min;
    double              CV_max;

    // Hyperparameters
    double              gradient_descent_stepsize;
    int                 batch_size;
    std::uint64_t       updates_count {0};
    // std::array<double, n_parameters>              gradient_descent_stepsizes{};
    // std::array<int,    n_parameters>              batch_sizes{};
    // std::array<std::uint64_t, n_parameters>       update_counts{};

    // Quadrature parameters
    double quadrature_abs_tol   {1e-8};
    double quadrature_rel_tol   {1e-6};
    int    quadrature_max_depth {15};

    std::vector<double> batch;
    // std::array<std::vector<double>, n_parameters> batches;

    struct MomentsT
    {
        ParametersT mean;
        std::array<std::array<double, 2>, 2> covariance;
    };

    // For the uniform target distribution, we can precompute the expectation values analytically
    // [[nodiscard]]
    // ParametersT ComputeUniformTargetDistExpectationValues() const noexcept
    // {
    //     const double c = BasisT::Constant();
    //     const double expectation_Q2   = (CV_max * CV_max + CV_min * CV_min + CV_max * CV_min)/ 3.0;

    //     const double expectation_sin2 = 0.5 - (std::sin(2.0 * c * CV_max) - std::sin(2.0 * c * CV_min)) / (4.0 * c * (CV_max - CV_min));

    //     return ParametersT{expectation_Q2, expectation_sin2};
    // }

    // 7-point Gauss, 15-point Kronrod
    template<typename FuncT>
    double GaussKronrodQuadrature(const FuncT& f, const double a, const double b, int depth = 0) const noexcept
    {
        static constexpr double nodes_kronrod[8]   = {0.991455371120813,
                                                      0.949107912342759,
                                                      0.864864423359769,
                                                      0.741531185599394,
                                                      0.586087235467691,
                                                      0.405845151377397,
                                                      0.207784955007898,
                                                      0.000000000000000};

        static constexpr double weights_kronrod[8] = {0.022935322010529,
                                                      0.063092092629979,
                                                      0.104790010322250,
                                                      0.140653259715525,
                                                      0.169004726639267,
                                                      0.190350578064785,
                                                      0.204432940075298,
                                                      0.209482141084728};

        static constexpr double weights_gauss[4] =   {0.129484966168870,
                                                      0.279705391489277,
                                                      0.381830050505119,
                                                      0.417959183673469};

        const double midpoint      = 0.5 * (a + b);
        const double half_interval = 0.5 * (b - a);

        const double f_midpoint = f(midpoint);
        double       I_kronrod  = weights_kronrod[7] * f_midpoint;
        double       I_gauss    = weights_gauss[3]   * f_midpoint;

        for (int i = 0; i < 7; ++i)
        {
            const double x_offset = half_interval * nodes_kronrod[i];
            const double f_left   = f(midpoint - x_offset);
            const double f_right  = f(midpoint + x_offset);
            const double sum      = f_left + f_right;

            I_kronrod += weights_kronrod[i] * sum;

            if (i == 1)
            {
                I_gauss += weights_gauss[0] * sum;
            }
            else if (i == 3)
            {
                I_gauss += weights_gauss[1] * sum;
            }
            else if (i == 5)
            {
                I_gauss += weights_gauss[2] * sum;
            }
        }

        I_kronrod *= half_interval;
        I_gauss   *= half_interval;

        const double tolerance = std::max(quadrature_abs_tol, quadrature_rel_tol * std::abs(I_kronrod));
        const double error     = std::abs(I_kronrod - I_gauss);

        if (error <= tolerance or depth >= quadrature_max_depth)
        {
            return I_kronrod;
        }
        return GaussKronrodQuadrature(f, a, midpoint, depth + 1) + GaussKronrodQuadrature(f, midpoint, b, depth + 1);
    }

    // For generic target distributions compute the expectation values of the basis functions using Gauss-Konrod
    [[nodiscard]]
    ParametersT ComputeTargetDistExpectationValues(const double a, const double b) const noexcept
    {
        const auto p  = [this](double x) noexcept
                        {
                            const double tmp = target_distribution(x);
                            return (std::isfinite(tmp) and tmp >= 0.0) ? tmp : 0.0;
                        };

        const auto f_0 = [this](double x) noexcept {return functional_basis.f0(x);};
        const auto f_1 = [this](double x) noexcept {return functional_basis.f1(x);};

        const double normalization = GaussKronrodQuadrature(p, a, b);

        // TODO: Or error?
        if (not std::isfinite(normalization) or normalization <= 0.0)
        {
            return ParametersT{0.0, 0.0};
        }

        const double I_0 = GaussKronrodQuadrature([&](double x) noexcept {return p(x) * f_0(x);}, a, b);
        const double I_1 = GaussKronrodQuadrature([&](double x) noexcept {return p(x) * f_1(x);}, a, b);

        return ParametersT{I_0 / normalization, I_1 / normalization};
    }

    [[nodiscard]]
    MomentsT ComputeBatchMoments(/*const std::vector<double>& batch*/) const noexcept
    {
        MomentsT moments{};

        std::array<double, n_parameters>                           mean{0.0, 0.0};
        std::array<std::array<double, n_parameters>, n_parameters> cov_matrix{{{0.0, 0.0}, {0.0, 0.0}}};

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

    // void UpdateAveragedParameter(int i) noexcept
    // {
    //     averaged_parameters[i] = (averaged_parameters[i] * update_counts[i] + functional_basis.parameters[i]) / static_cast<double>(update_counts[i] + 1);
    //     update_counts[i]      += 1;
    // }

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
        const MomentsT    batch_moments = ComputeBatchMoments();
        const ParametersT gradient      = ParametersT{-batch_moments.mean[0] + target_distribution_expectation_values[0],
                                                                  -batch_moments.mean[1] + target_distribution_expectation_values[1]};

        const double diff_0 = functional_basis.parameters[0] - averaged_parameters[0];
        const double diff_1 = functional_basis.parameters[1] - averaged_parameters[1];

        // Update with averaged stochastic gradient descent algorithm, using the full Hessian in contrast to the original VES paper (where only the diagonals are used)
        functional_basis.parameters[0] -= gradient_descent_stepsize * (gradient[0] + batch_moments.covariance[0][0] * diff_0 + batch_moments.covariance[0][1] * diff_1);
        functional_basis.parameters[1] -= gradient_descent_stepsize * (gradient[1] + batch_moments.covariance[1][0] * diff_0 + batch_moments.covariance[1][1] * diff_1);

        UpdateAveragedParameters();

        batch.clear();
    }
public:

    VariationalBiasPotential(const BasisT& functional_basis_in, const TargetDistT& target_distribution_in, const double CV_min_in, const double CV_max_in, const double gradient_descent_stepsize_in, const int batch_size_in) :
    functional_basis(functional_basis_in),
    target_distribution(target_distribution_in),
    // averaged_parameters(BasisT::ParametersT.),
    CV_min(CV_min_in),
    CV_max(CV_max_in),
    gradient_descent_stepsize(gradient_descent_stepsize_in),
    batch_size(batch_size_in)
    {
        // averaged_parameters.assign(static_cast<std::size_t>(batch_size), 0.0);
        averaged_parameters = functional_basis.parameters;
        // This relies on several members already being initialized, so can not use it in the member initializer list above
        target_distribution_expectation_values = ComputeTargetDistExpectationValues(CV_min, CV_max);
        std::cout << "\nInitialized VariationalBiasPotential with the following parameters:\n"
                  << "  functional_basis_name:     " << BasisT::GetName()         << "\n"
                  << "  target_distribution_name:  " << TargetDistT::GetName()    << "\n"
                  << "  CV_min:                    " << CV_min                    << "\n"
                  << "  CV_max:                    " << CV_max                    << "\n"
                  << "  gradient_descent_stepsize: " << gradient_descent_stepsize << "\n"
                  << "  batch_size:                " << batch_size                << "\n" << std::endl;
    }

    // VariationalBiasPotential(const BasisT& functional_basis_in, const double CV_min_in, const double CV_max_in, const std::array<double, n_parameters>& gradient_descent_stepsizes_in, const std::array<int, n_parameters> batch_sizes_in) :
    // functional_basis(functional_basis_in),
    // // averaged_parameters(BasisT::ParametersT.),
    // CV_min(CV_min_in),
    // CV_max(CV_max_in),
    // gradient_descent_stepsize(gradient_descent_stepsize_in),
    // batch_size(batch_size_in)
    // {
    //     // averaged_parameters.assign(static_cast<std::size_t>(batch_size), 0.0);
    //     averaged_parameters = functional_basis.parameters;
    //     // This relies on several members already being initialized, so can not use it in the member initializer list above
    //     target_distribution_expectation_values = ComputeTargetDistExpectationValues(CV_min, CV_max);
    //     std::cout << "\nInitialized VariationalBiasPotential with the following parameters:\n"
    //               << "  functional_basis_name:     " << BasisT::GetName()                                                      << "\n"
    //               << "  target_distribution_name:  " << TargetDistT::GetName()                                                 << "\n"
    //               << "  CV_min:                    " << CV_min                                                                 << "\n"
    //               << "  CV_max:                    " << CV_max                                                                 << "\n"
    //               << "  gradient_descent_stepsize: " << gradient_descent_stepsizes[0] << ", " << gradient_descent_stepsizes[1] << "\n"
    //               << "  batch_size:                " << batch_sizes[0]                << ", " << gradient_descent_stepsizes[1] << "\n" << std::endl;
    // }

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

    // Only provided for compatibility with Metadynamics code
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
        // Use averaged parameter values
        BasisT tmp_basis     = functional_basis;
        tmp_basis.parameters = averaged_parameters;
        return tmp_basis.Evaluate(CV);
        // return functional_basis.Evaluate(CV);
    }

    [[nodiscard]]
    double ReturnDerivative(const double CV) const noexcept
    {
        // Use averaged parameter values
        BasisT tmp_basis     = functional_basis;
        tmp_basis.parameters = averaged_parameters;
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
            << "VES potential"               << "\n"
            << "basis_name: "                << BasisT::GetName()              << "\n"
            << "target_name: "               << TargetDistT::GetName()         << "\n"
            << "CV_min: "                    << CV_min                         << "\n"
            << "CV_max: "                    << CV_max                         << "\n"
            << "gradient_descent_stepsize: " << gradient_descent_stepsize      << "\n"
            << "batch_size: "                << batch_size                     << "\n"
            << "alpha_1: "                   << functional_basis.parameters[0] << "\n"
            << "alpha_2: "                   << functional_basis.parameters[1] << "\n"
            << "alpha_1_avg: "               << averaged_parameters[0]         << "\n"
            << "alpha_2_avg: "               << averaged_parameters[1]         << "\n"
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

        ofs << "alpha_1: "                   << functional_basis.parameters[0] << "\n"
            << "alpha_2: "                   << functional_basis.parameters[1] << "\n"
            << "alpha_1_avg: "               << averaged_parameters[0]         << "\n"
            << "alpha_2_avg: "               << averaged_parameters[1]         << "\n";
        // TODO: Add loss function/VES functional
            // << "Loss: "                      << ComputeLossFunction() << "\n";
        const double grid_distance = (CV_max - CV_min) / static_cast<double>(grid_points - 1);
        for (int i = 0; i < grid_points - 1; ++i)
        {
            ofs << ReturnPotential(CV_min + grid_distance * i) << ",";
        }
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
