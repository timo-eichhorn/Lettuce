#ifndef LETTUCE_METADYNAMICS_HPP
#define LETTUCE_METADYNAMICS_HPP

// Non-standard library headers
#include "IO/ansi_colors.hpp"
#include "IO/string_manipulation.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
//----------------------------------------
// Standard C headers
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

class MetaBiasPotential
{
private:
    std::vector<double> bias_grid{};
    double              CV_current{0.0};
    double              CV_min;
    double              CV_max;
    int                 bin_number;
    int                 grid_point_number;
    double              bin_width;
    double              bin_width_inverse;
    double              weight;
    double              well_tempered_parameter;
    double              threshold_weight;
    std::uint_fast64_t  exceeded_count{0};

    [[nodiscard]]
    int BinIndexFromCV(const double CV) const noexcept
    {
        return static_cast<int>(std::floor((CV - CV_min) * bin_width_inverse));
    }

    [[nodiscard]]
    double CVFromBinIndex(const int index) const noexcept
    {
        return CV_min + index * bin_width;
    }
public:
    MetaBiasPotential(const double CV_min_in, const double CV_max_in, const int bin_number_in, const double weight_in, const double well_tempered_parameter_in, const double threshold_weight_in) :
    CV_min(CV_min_in),
    CV_max(CV_max_in),
    bin_number(bin_number_in),
    grid_point_number(bin_number_in + 1),
    bin_width((CV_max - CV_min) / bin_number),
    bin_width_inverse(1.0 / bin_width),
    weight(weight_in),
    well_tempered_parameter(well_tempered_parameter_in),
    threshold_weight(threshold_weight_in)
    {
        if (CV_min >= CV_max)
        {
            throw std::invalid_argument("CV_min < CV_max required");
        }
        if (bin_number <= 0)
        {
            throw std::invalid_argument("bin_number > 0 required");
        }
        bias_grid.assign(static_cast<std::size_t>(grid_point_number), 0.0);
        std::cout << "\nInitialized MetaBiasPotential with the following parameters:\n"
                  << "  CV_min:                  " << CV_min                  << "\n"
                  << "  CV_max:                  " << CV_max                  << "\n"
                  << "  bin_number:              " << bin_number              << "\n"
                  << "  grid_point_number:       " << grid_point_number       << "\n"
                  << "  bin_width:               " << bin_width               << "\n"
                  << "  bin_width_inverse:       " << bin_width_inverse       << "\n"
                  << "  weight:                  " << weight                  << "\n"
                  << "  well_tempered_parameter: " << well_tempered_parameter << "\n"
                  << "  threshold_weight:        " << threshold_weight        << "\n"
                  << "  exceeded_count:          " << exceeded_count          << "\n" << std::endl;
    }

    template<typename FuncT>
    void GeneratePotentialFrom(FuncT&& generator_function) noexcept
    {
        for (std::size_t bin_index = 0; bin_index < bias_grid.size(); ++bin_index)
        {
            bias_grid[bin_index] = static_cast<double>(generator_function(CVFromBinIndex(bin_index)));
        }
    }

    // TODO: Introduce adjustable Gaussian variance that is independent of the bin width?
    //       I suspect the Gaussians are currently too narrow
    // TODO: Could truncate updates after a certain distance from CV value due to exponential fall-off
    void UpdatePotential(const double CV) noexcept
    {
        // Gaussian histogram, handle points outside the range [CV_min, CV_max) like regular points, but track via exceeded_count
        if (CV < CV_min or CV >= CV_max)
        {
            exceeded_count += 1;
        }
        for (std::size_t bin = 0; bin < bias_grid.size(); ++bin)
        {
            const double CV_current_bin {CVFromBinIndex(bin)};
            const double dist           {CV - CV_current_bin};
            if constexpr(metapotential_well_tempered)
            {
                bias_grid[bin] += weight * std::exp(-0.5 * bin_width_inverse * bin_width_inverse * dist * dist - bias_grid[bin] / well_tempered_parameter);
            }
            else
            {
                bias_grid[bin] += weight * std::exp(-0.5 * bin_width_inverse * bin_width_inverse * dist * dist);
            }
        }
    }

    void UpdatePotential(const std::vector<double>& CV_vec) noexcept
    {
        for (const double CV : CV_vec)
        {
            UpdatePotential(CV);
        }
    }

    void UpdatePotentialSymmetric(const double CV) noexcept
    {
        UpdatePotential( CV);
        UpdatePotential(-CV);
    }

    void UpdatePotentialSymmetric(const std::vector<double>& CV_vec) noexcept
    {
        for (const double CV : CV_vec)
        {
            UpdatePotential( CV);
            UpdatePotential(-CV);
        }
    }

    // Linear interpolation of potential in the interval [CV_min, CV_max)
    // Outside the range, the closest edge value plus a quadratic penalty term is returned
    [[nodiscard]]
    double ReturnPotential(const double CV) const noexcept
    {
        // Get the index to the left of the current CV value
        const int bin_index {BinIndexFromCV(CV)};

        // Casting to unsigned int means we check for values that are in the range [0, grid_point_number - 1)
        // Values equal to grid_point_number - 1 are not allowed, since we interpolate between the bin_index and bin_index + 1
        // Effectively means that our potential is limited to the interval [CV_min, CV_max)
        if (static_cast<unsigned int>(bin_index) < static_cast<unsigned int>(grid_point_number - 1))
        {
            double interpolation_constant {(CV - CVFromBinIndex(bin_index)) * bin_width_inverse};
            return bias_grid[bin_index] * (1.0 - interpolation_constant) + interpolation_constant * bias_grid[bin_index + 1];
        }
        else
        {
            // If we fall outside the defined range, return the edge value and an additional quadratic penalty term
            if (bin_index < 0)
            {
                const double dist = CV - CV_min;
                return bias_grid.front() + threshold_weight * dist * dist;
            }
            else
            {
                const double dist = CV - CV_max;
                return bias_grid.back()  + threshold_weight * dist * dist;
            }
        }
    }

    [[nodiscard]]
    double ReturnDerivative(const double CV) const noexcept
    {
        // Use symmetric finite difference
        const double h = 0.5 * bin_width;
        return bin_width_inverse * (ReturnPotential(CV + h) - ReturnPotential(CV - h));
    }

    void SymmetrizePotential() noexcept
    {
        if (bias_grid.empty())
        {
            return;
        }
        std::size_t i = 0;
        std::size_t j = bias_grid.size() - 1;
        while (i < j)
        {
            const double avg_val = 0.5 * (bias_grid[i] + bias_grid[j]);
            bias_grid[i] = bias_grid[j] = avg_val;
            ++i;
            --j;
        }
        std::cout << "Bias potential symmetrized (average value)!" << std::endl;
    }

    // Symmetrize by taking the maximum value instead of the average value
    void SymmetrizePotentialMaximum() noexcept
    {
        if (bias_grid.empty())
        {
            return;
        }
        std::size_t i = 0;
        std::size_t j = bias_grid.size() - 1;
        while (i < j)
        {
            const double max_val = std::max(bias_grid[i], bias_grid[j]);
            bias_grid[i] = bias_grid[j] = max_val;
            ++i;
            --j;
        }
        std::cout << "Bias potential symmetrized (max value)!" << std::endl;
    }

    // Create a penalty weight for values below CV_lower and values above CV_upper, and write the penalty potential into a file
    [[deprecated]]
    void AddPenaltyWeight(const double CV_lower, const double CV_upper, const std::string& filename) noexcept
    {
        assert(CV_lower < CV_upper);
        std::vector<double> penalty_potential(bias_grid.size(), 0.0);
        const int lower_index {BinIndexFromCV(CV_lower)};
        const int upper_index {BinIndexFromCV(CV_upper)};

        for (int ind = 0; ind < lower_index; ++ind)
        {
            const double CV   {CVFromBinIndex(ind)};
            const double dist {CV - CV_lower};
            penalty_potential[ind] += threshold_weight * dist * dist;
        }
        for (int ind = upper_index; ind < grid_point_number; ++ind)
        {
            const double CV   {CVFromBinIndex(ind)};
            const double dist {CV - CV_upper};
            penalty_potential[ind] += threshold_weight * dist * dist;
        }
        std::transform(bias_grid.begin(), bias_grid.end(), penalty_potential.begin(), bias_grid.begin(), std::plus<double>());

        std::ofstream ofs;
        ofs.open(filename, std::fstream::out | std::fstream::app);
        std::copy(penalty_potential.cbegin(), std::prev(penalty_potential.cend()), std::ostream_iterator<double>(ofs, ","));
        ofs << penalty_potential.back() << "\n";
    }

    [[deprecated]]
    void SubtractPenaltyWeight(const double CV_lower, const double CV_upper) noexcept
    {
        assert(CV_lower < CV_upper);
        const int lower_index {BinIndexFromCV(CV_lower)};
        const int upper_index {BinIndexFromCV(CV_upper)};

        for (int ind = 0; ind < lower_index; ++ind)
        {
            const double CV   {CVFromBinIndex(ind)};
            const double dist {CV - CV_lower};
            bias_grid[ind] -= threshold_weight * dist * dist;
        }
        for (int ind = upper_index; ind < grid_point_number; ++ind)
        {
            const double CV   {CVFromBinIndex(ind)};
            const double dist {CV - CV_upper};
            bias_grid[ind] -= threshold_weight * dist * dist;
        }
    }

    void SetCV_current(const double CV_in) noexcept
    {
        CV_current = CV_in;
    }

    void SetWeight(const double weight_in) noexcept
    {
        weight = weight_in;
    }

    [[nodiscard]]
    double ReturnCV_current() const noexcept
    {
        return CV_current;
    }

    [[nodiscard]]
    double ReturnBinWidth() const noexcept
    {
        return bin_width;
    }

    [[nodiscard]]
    double ReturnBinWidthInverse() const noexcept
    {
        return bin_width_inverse;
    }

    void SaveParameters(const std::string& filename, const bool overwrite = false)
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
        ofs << program_version                                        << "\n"
            << "Metadynamics Potential"                               << "\n"
            << "CV_min: "                  << CV_min                  << "\n"
            << "CV_max: "                  << CV_max                  << "\n"
            << "bin_number: "              << bin_number              << "\n"
            << "weight: "                  << weight                  << "\n"
            << "well_tempered_parameter: " << well_tempered_parameter << "\n"
            << "threshold_weight: "        << threshold_weight        << "\n"
            << "END_METADYN_PARAMS"                                   << "\n";

    }

    void SavePotential(const std::string& filename, const bool overwrite = false)
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
        // This shouldn't happen
        if (bias_grid.empty())
        {
            ofs << "\nexceeded_count: " << exceeded_count << "\n";
            return;
        }
        std::copy(bias_grid.cbegin(), std::prev(bias_grid.cend()), std::ostream_iterator<double>(ofs, ","));
        ofs << bias_grid.back()                     << "\n"
            << "exceeded_count: " << exceeded_count << "\n";
    }

    // Function that loads the histogram parameters and the histogram itself from a file
    // TODO: Work with string_view instead of strings?
    // TODO: This seems to be quite slow: https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring#comment5920160_2602060
    //       Better use stringstreams?
    bool LoadPotential(const std::string& filename)
    {
        if (!std::filesystem::exists(filename))
        {
            std::cerr << Lettuce::Color::BoldRed << "File " << filename << " not found!" << Lettuce::Color::Reset << "\n";
            return false;
        }

        std::ifstream ifs;
        ifs.open(filename, std::fstream::in);
        if (!ifs)
        {
            std::cerr << Lettuce::Color::BoldRed << "Reading potential from file " << filename << " failed!" << Lettuce::Color::Reset << "\n";
            return false;
        }

        // Count the total line number using std::count (add one to the final result since we only counted the number of '\n' characters) and return to beginning of file
        std::size_t linecount = std::count(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>(), '\n') + 1;
        ifs.clear();
        ifs.seekg(0, ifs.beg);

        // Start reading parameters
        std::string current_line;
        std::getline(ifs, current_line);
        if (current_line != program_version)
        {
            std::cerr << Lettuce::Color::BoldRed << "Metadynamics potential file comes from an incompatible program version!\n"
                                                 << "Current version: " << program_version << "\n"
                                                 << "File version:    " << current_line    << "\n" << Lettuce::Color::Reset;
            return false;
        }

        std::cout << "Loading metadynamics potential from: " << filename << "\n";
        std::cout << "The file has " << linecount << " lines in total." << std::endl;

        bool end_token_found {false};
        while(std::getline(ifs, current_line))
        {
            // Search for parameters until "END_METADYN_PARAMS"
            if (current_line.find("END_METADYN_PARAMS") != std::string::npos)
            {
                end_token_found = true;
                break;
            }
            std::size_t pos {std::string::npos};
            // Get CV_min
            if ((pos = FindTokenEnd(current_line, "CV_min: ")) != std::string::npos)
            {
                CV_min = std::stod(current_line.substr(pos));
                continue;
            }
            // Get CV_max
            if ((pos = FindTokenEnd(current_line, "CV_max: ")) != std::string::npos)
            {
                CV_max = std::stod(current_line.substr(pos));
                continue;
            }
            // Get bin_number
            if ((pos = FindTokenEnd(current_line, "bin_number: ")) != std::string::npos)
            {
                bin_number = std::stoi(current_line.substr(pos));
                continue;
            }
            // Get weight
            if ((pos = FindTokenEnd(current_line, "weight: ")) != std::string::npos)
            {
                weight = std::stod(current_line.substr(pos));
                continue;
            }
            // Get well_tempered_parameter
            if ((pos = FindTokenEnd(current_line, "well_tempered_parameter: ")) != std::string::npos)
            {
                well_tempered_parameter = std::stod(current_line.substr(pos));
                continue;
            }
            // Get threshold_weight
            if ((pos = FindTokenEnd(current_line, "threshold_weight: ")) != std::string::npos)
            {
                threshold_weight = std::stod(current_line.substr(pos));
                continue;
            }
        }
        // Warn, but still proceed if 'END_METADYN_PARAMS' is not found
        if (!end_token_found)
        {
            std::cerr << Lettuce::Color::BoldRed << "END_METADYN_PARAMS not found!" << Lettuce::Color::Reset << "\n";
        }

        // Calculate secondary parameters from primary parameters
        if (CV_min >= CV_max)
        {
            std::cerr << "Invalid CV_min and CV_max combination read from file\n";
            return false;
        }
        if (bin_number <= 0)
        {
            std::cerr << "Invalid bin_number read from file\n";
            return false;
        }
        grid_point_number = bin_number + 1;
        bin_width         = (CV_max - CV_min) / bin_number;
        bin_width_inverse = 1.0 / bin_width;

        // Iterate to the last two lines of the file (linecount - 3, since we still want to read in the second to last line)
        ifs.clear();
        ifs.seekg(0, ifs.beg);
        for (std::size_t ind = 0; ind < linecount - 3; ++ind)
        {
            ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        // Load histogram into bias_grid
        bias_grid.clear();
        bias_grid.reserve(grid_point_number);
        std::getline(ifs, current_line);
        if (not current_line.empty() and current_line.back() == '\r')
        {
            current_line.pop_back();
        }
        std::size_t pos = 0;
        while (pos < current_line.size())
        {
            const auto next_pos = current_line.find(",", pos);
            bias_grid.push_back(std::stod(current_line.substr(pos, next_pos - pos)));
            pos = (next_pos == std::string::npos ? current_line.size() : next_pos + 1);
        }

        // Get exceeded count
        std::getline(ifs, current_line);
        EraseUntil(current_line, "exceeded_count: ", true);
        exceeded_count = std::stoull(current_line);

        // Print message with parameters
        std::cout << "\nSuccessfully loaded MetaBiasPotential from " << filename << " with the following parameters:\n"
                  << "  CV_min:                  " << CV_min                  << "\n"
                  << "  CV_max:                  " << CV_max                  << "\n"
                  << "  bin_number:              " << bin_number              << "\n"
                  << "  grid_point_number:       " << grid_point_number       << "\n"
                  << "  bin_width:               " << bin_width               << "\n"
                  << "  bin_width_inverse:       " << bin_width_inverse       << "\n"
                  << "  weight:                  " << weight                  << "\n"
                  << "  well_tempered_parameter: " << well_tempered_parameter << "\n"
                  << "  threshold_weight:        " << threshold_weight        << "\n"
                  << "  exceeded_count:          " << exceeded_count          << "\n" << std::endl;
        return true;
    }
};

#endif // LETTUCE_METADYNAMICS_HPP
