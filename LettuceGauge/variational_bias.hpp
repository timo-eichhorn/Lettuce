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
#include <format>
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

// TODO: Currently handling of ParametersT suboptimal...
namespace Optimizers
{
    template<typename ParametersT>
    struct UpdateInfo
    {
        ParametersT gradient{};
        ParametersT difference_to_average{};

        // TODO: Hard-coded for now
        std::array<std::array<double, 2>, 2> covariance{};
    };

    template<typename ParametersT>
    struct AveragedStochasticGradientDescent
    {
        double        stepsize;
        double        momentum;
        int           batch_size;

        ParametersT   velocity{};
        // TODO: Used for Polyak averaging, might replace with exponentially decaying moving average later
        std::uint64_t updates_count{};

        AveragedStochasticGradientDescent(double stepsize_in, double momentum_in, int batch_size_in) : stepsize(stepsize_in), momentum(momentum_in), batch_size(batch_size_in) {}

        [[nodiscard]]
        int GetBatchSize() const noexcept
        {
            return batch_size;
        }

        void Update(ParametersT& parameters, ParametersT& averaged_parameters, const UpdateInfo<ParametersT>& update_info) noexcept
        {
            // TODO: Currently still uses hard-coded number of parameters
            ParametersT delta{};

            // Update with averaged stochastic gradient descent algorithm, using the full Hessian in contrast to the original VES paper (where only the diagonals are used)
            delta[0] = update_info.gradient[0] + update_info.covariance[0][0] * update_info.difference_to_average[0] + update_info.covariance[0][1] * update_info.difference_to_average[1];
            delta[1] = update_info.gradient[1] + update_info.covariance[1][0] * update_info.difference_to_average[0] + update_info.covariance[1][1] * update_info.difference_to_average[1];

            velocity[0] = momentum * velocity[0] + stepsize * delta[0];
            velocity[1] = momentum * velocity[1] + stepsize * delta[1];

            parameters[0] -= velocity[0];
            parameters[1] -= velocity[1];

            for (std::size_t i = 0; i < parameters.size(); ++i)
            {
                // Polyak averaging
                // averaged_parameters[i] = (averaged_parameters[i] * static_cast<double>(updates_count) + parameters[i]) / static_cast<double>(updates_count + 1);
                // Exponentially decaying moving average
                const double decay_factor = 0.9;
                averaged_parameters[i]    = decay_factor * averaged_parameters[i] + (1.0 - decay_factor) * parameters[i];
            }
            updates_count += 1;
        }

        [[nodiscard]]
        static std::string GetName()
        {
            return "AveragedStochasticGradientDescent";
        }

        [[nodiscard]]
        std::string GetParameters() const
        {
            return std::format("\nstepsize: {}\nmomentum: {}\nbatch_size (current): {}", stepsize, momentum, batch_size);
        }
    };

    template<typename ParametersT>
    struct Adam
    {
        // Default parameters from original paper [1412.6980]
        double        alpha   {0.001};
        double        beta_1  {0.9};
        double        beta_2  {0.999};
        double        epsilon {1.e-8};
        std::uint64_t t       {0u};
        int           batch_size;

        ParametersT   m{}; // First moment vector
        ParametersT   v{}; // Second moment vector

        explicit Adam(int batch_size_in) : batch_size(batch_size_in) {}

        [[nodiscard]]
        int GetBatchSize() const noexcept
        {
            return batch_size;
        }

        // TODO: The update_info member covariance and difference_to_average are unused here
        void Update(ParametersT& parameters, ParametersT& averaged_parameters, const UpdateInfo<ParametersT>& update_info) noexcept
        {
            t += 1;
            const double t_d = static_cast<double>(t);

            for (std::size_t i = 0; i < m.size(); ++i)
            {
                // TODO: Compute or get gradient
                m[i] = beta_1 * m[i] + (1.0 - beta_1) * update_info.gradient[i];
                v[i] = beta_2 * v[i] + (1.0 - beta_2) * update_info.gradient[i] * update_info.gradient[i];

                // Inefficient version
                // const double m_hat = m[i] / (1.0 - std::pow(beta_1, t_d));
                // const double v_hat = v[i] / (1.0 - std::pow(beta_2, t_d));
                // parameters[i] -= alpha * m_hat / (std::sqrt(v_hat) + epsilon);
                // More efficient version
                const double alpha_t = alpha * std::sqrt(1.0 - std::pow(beta_2, t_d)) / (1.0 - std::pow(beta_1, t_d));
                parameters[i] -= alpha_t * m[i] / (std::sqrt(v[i]) + epsilon);
            }

            for (std::size_t i = 0; i < parameters.size(); ++i)
            {
                // Polyak averaging
                averaged_parameters[i] = (averaged_parameters[i] * (t_d - 1.0) + parameters[i]) / t_d;
                // Exponentially decaying moving average
                const double decay_factor = 0.9;
                averaged_parameters[i]    = decay_factor * averaged_parameters[i] + (1.0 - decay_factor) * parameters[i];
            }
        }

        [[nodiscard]]
        static std::string GetName()
        {
            return "Adam";
        }

        [[nodiscard]]
        std::string GetParameters() const
        {
            return std::format("\nalpha: {}\nbeta_1: {}\nbeta_2: {}\nepsilon: {}\nbatch_size (current): {}", alpha, beta_1, beta_2, epsilon, batch_size);
        }
    };
} // namespace Optimizers

template<typename BasisT, typename TargetDistT, typename OptimizerT>
struct VariationalBiasPotential
{
public:
    using ParametersT = BasisT::ParametersT;
    static constexpr int n_parameters = BasisT::n_parameters;
private:
    BasisT              functional_basis;
    TargetDistT         target_distribution;
    OptimizerT          optimizer;

    ParametersT         averaged_parameters;
    ParametersT         target_distribution_expectation_values;

    double              CV_current               {0.0};
    double              CV_min;
    double              CV_max;
    double              CV_current_min;
    double              CV_current_max;
    double              CV_domain_padding_factor {1.15};
    double              CV_max_abs_observed      {0.0};
    const double        penalty_prefactor        {100.0};
    const double        penalty_power            {4.0};
    int                 batch_size;

    // Quadrature parameters
    double              quadrature_abs_tol       {1e-8};
    double              quadrature_rel_tol       {1e-6};
    int                 quadrature_max_depth     {15};

    std::vector<double> batch;

    struct MomentsT
    {
        ParametersT mean;
        std::array<std::array<double, 2>, 2> covariance;
    };

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

        static constexpr double weights_gauss[4]   = {0.129484966168870,
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
        const auto p   = [this](double x) noexcept
                         {
                             const double tmp = target_distribution(x);
                             return (std::isfinite(tmp) and tmp >= 0.0) ? tmp : 0.0;
                         };

        const double normalization = GaussKronrodQuadrature(p, a, b);

        // TODO: Or error?
        if (not std::isfinite(normalization) or normalization <= 0.0)
        {
            return ParametersT{0.0, 0.0};
        }

        const auto f_0 = [this](double x) noexcept {return functional_basis.f0(x);};
        const auto f_1 = [this](double x) noexcept {return functional_basis.f1(x);};

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

        // Compute covariance matrix in a single pass using Welford's algorithm
        // TODO: This is not necessary for most (first-order) optimizers like Adam
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

    // Assumes a domain and CV distribution that is symmetric around 0
    void MaybeExpandDomain(const double CV) noexcept
    {
        const double abs_CV = std::abs(CV);

        if (abs_CV <= CV_max_abs_observed)
        {
            return;
        }

        CV_max_abs_observed = abs_CV;
        // const double CV_limit         = std::max(std::abs(CV_min), std::abs(CV_max));
        const double current_limit  = std::max(std::abs(CV_current_min), std::abs(CV_current_max));
        const double max_limit      = std::min(std::abs(CV_min), std::abs(CV_max));
        // const double proposed_limit = std::clamp(CV_domain_padding_factor * current_limit, CV_min, CV_max);
        // Base new limit on observed value instead of the old limit
        double proposed_limit = std::max(current_limit, CV_domain_padding_factor * abs_CV);
        proposed_limit        = std::min(proposed_limit, max_limit);

        // TODO: Add minimum change to update?
        if (proposed_limit > current_limit)
        {
            CV_current_min = -proposed_limit;
            CV_current_max =  proposed_limit;
            // Update target distribution expectation values
            target_distribution_expectation_values = ComputeTargetDistExpectationValues(CV_current_min, CV_current_max);
            // TODO: Assuming a random walk on the domain, we would need to rescale the batch size proportionally to the squared domain size
        }
    }

    void MaybeUpdate() noexcept
    {
        if (static_cast<int>(batch.size()) < batch_size)
        {
            return;
        }
        const MomentsT    batch_moments = ComputeBatchMoments();
        const ParametersT gradient{-batch_moments.mean[0] + target_distribution_expectation_values[0],
                                   -batch_moments.mean[1] + target_distribution_expectation_values[1]};

        const ParametersT difference_to_average{functional_basis.parameters[0] - averaged_parameters[0],
                                                functional_basis.parameters[1] - averaged_parameters[1]};


        Optimizers::UpdateInfo<ParametersT> info{gradient, difference_to_average, batch_moments.covariance};

        optimizer.Update(functional_basis.parameters, averaged_parameters, info);

        batch.clear();
    }

    [[nodiscard]]
    double ReturnPenaltyTerm(const double CV) const noexcept
    {
        if (CV < CV_min)
        {
            const double dist = CV_min - CV;
            return penalty_prefactor * std::pow(dist, penalty_power);
        }
        if (CV > CV_max)
        {
            const double dist = CV - CV_max;
            return penalty_prefactor * std::pow(dist, penalty_power);
        }
        return 0.0;
    }

    [[nodiscard]]
    double ReturnPenaltyTermDerivative(const double CV) const noexcept
    {
        if (CV < CV_min)
        {
            const double dist = CV_min - CV;
            return -penalty_power * penalty_prefactor * std::pow(dist, penalty_power - 1);
        }
        if (CV > CV_max)
        {
            const double dist = CV - CV_max;
            return penalty_power * penalty_prefactor * std::pow(dist, penalty_power - 1);
        }
        return 0.0;
    }
public:

    VariationalBiasPotential(const BasisT& functional_basis_in, const TargetDistT& target_distribution_in, const OptimizerT& optimizer_in, const double CV_min_in, const double CV_max_in, double CV_current_min_in, double CV_current_max_in, const int batch_size_in) :
    functional_basis(functional_basis_in),
    target_distribution(target_distribution_in),
    optimizer(optimizer_in),
    averaged_parameters(functional_basis_in.parameters),
    CV_min(CV_min_in),
    CV_max(CV_max_in),
    CV_current_min(CV_current_min_in),
    CV_current_max(CV_current_max_in),
    batch_size(batch_size_in)
    {
        // averaged_parameters.assign(static_cast<std::size_t>(batch_size), 0.0);
        // averaged_parameters = functional_basis.parameters;
        // This relies on several members already being initialized, so can not use it in the member initializer list above
        // target_distribution_expectation_values = ComputeTargetDistExpectationValues(CV_min, CV_max);
        target_distribution_expectation_values = ComputeTargetDistExpectationValues(CV_current_min, CV_current_max);
        std::cout << "\nInitialized VariationalBiasPotential with the following parameters:\n"
                  << "  functional_basis_name:     " << BasisT::GetName()         << "\n"
                  << "  target_distribution_name:  " << TargetDistT::GetName()    << "\n"
                  << "  optimizer_name:            " << OptimizerT::GetName()     << "\n"
                  << "  optimizer_parameters:      " << optimizer.GetParameters() << "\n"
                  << "  CV_min:                    " << CV_min                    << "\n"
                  << "  CV_max:                    " << CV_max                    << "\n"
                  << "  CV_current_min:            " << CV_current_min            << "\n"
                  << "  CV_current_max:            " << CV_current_max            << "\n" << std::endl;
                  // << "  batch_size:                " << batch_size                << "\n" << std::endl;
    }

    void UpdatePotential(const double CV) noexcept
    {
        MaybeExpandDomain(CV);
        batch.push_back(CV);
        MaybeUpdate();
    }

    void UpdatePotential(const std::vector<double>& CV_vec) noexcept
    {
        for (double CV : CV_vec)
        {
            MaybeExpandDomain(CV);
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

    // TODO: Either manually include a penalty term here, or move to functional basis?
    //       Want to explicitly constrain the simulation to stay inside the specified domain
    [[nodiscard]]
    double ReturnPotential(const double CV) const noexcept
    {
        // Use averaged parameter values
        BasisT tmp_basis     = functional_basis;
        tmp_basis.parameters = averaged_parameters;
        return tmp_basis.Evaluate(CV) + ReturnPenaltyTerm(CV);
    }

    [[nodiscard]]
    double ReturnDerivative(const double CV) const noexcept
    {
        // Use averaged parameter values
        BasisT tmp_basis     = functional_basis;
        tmp_basis.parameters = averaged_parameters;
        return tmp_basis.Derivative(CV) + ReturnPenaltyTermDerivative(CV);
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
            << "optimizer_name: "            << OptimizerT::GetName()          << "\n"
            << "optimizer_parameters: "      << optimizer.GetParameters()      << "\n"
            << "CV_min: "                    << CV_min                         << "\n"
            << "CV_max: "                    << CV_max                         << "\n"
            << "CV_current_min: "            << CV_current_min                 << "\n"
            << "CV_current_max: "            << CV_current_max                 << "\n"
            << "penalty_prefactor: "         << penalty_prefactor              << "\n"
            << "penalty_power: "             << penalty_power                  << "\n"
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
            << "alpha_2_avg: "               << averaged_parameters[1]         << "\n"
            << "CV_current_min: "            << CV_current_min                 << "\n"
            << "CV_current_max: "            << CV_current_max                 << "\n";
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
