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
#include <iostream>
#include <iterator>
#include <string>
// #include <string_view>
#include <vector>
//----------------------------------------
// Standard C headers
#include <cassert>
#include <cmath>
#include <cstddef>

class MetaBiasPotential
{
private:
    std::vector<double> bin_count;
    double CV_current;
    double CV_min, CV_max;
    int bin_number;
    int edge_number;
    double bin_width, bin_width_inverse, weight, well_tempered_parameter, threshold_weight;
    uint_fast64_t exceeded_count;
    std::ofstream binlog;
    std::ifstream binload;
public:
    // TODO: Rewrite constructor
    // MetaBiasPotential(const double CV_min_in, const double CV_max_in, const size_t bin_number_in, const double weight_in, const double well_tempered_parameter_in, const double threshold_weight_in) noexcept :
    // CV_min(CV_min_in), CV_max(CV_max_in), bin_number(bin_number_in), edge_number(bin_number_in + 1) weight(weight_in), well_tempered_parameter(well_tempered_parameter_in), threshold_weight(threshold_weight_in)
    // {
    //     assert(CV_min_in < CV_max_in);
    //     // Resize bin_count and set all entries to 0
    //     bin_count.assign(edge_number, 0.0);
    //     // Reset out of range count
    //     exceeded_count = 0;
    //     // Print message with parameters
    //     std::cout << "\nInitialized MetaBiasPotential with the following parameters:\n";
    //     std::cout << "  CV_min: " << CV_min << "\n";
    //     std::cout << "  CV_max: " << CV_max << "\n";
    //     std::cout << "  bin_number: " << bin_number << "\n";
    //     std::cout << "  edge_number: " << edge_number << "\n";
    //     std::cout << "  bin_width: " << bin_width << "\n";
    //     std::cout << "  bin_width_inverse: " << bin_width_inverse << "\n";
    //     std::cout << "  weight: " << weight << "\n";
    //     std::cout << "  threshold_weight: " << threshold_weight << "\n";
    //     std::cout << "  exceeded_count: " << exceeded_count << "\n" << std::endl;
    // }
    MetaBiasPotential(const double CV_min_in, const double CV_max_in, const int bin_number_in, const double weight_in, const double well_tempered_parameter_in, const double threshold_weight_in)
    {
        assert(CV_min_in < CV_max_in);
        CV_min                  = CV_min_in;
        CV_max                  = CV_max_in;
        bin_number              = bin_number_in;
        edge_number             = bin_number + 1;
        bin_width               = (CV_max - CV_min) / bin_number;
        bin_width_inverse       = 1.0 / bin_width;
        weight                  = weight_in;
        well_tempered_parameter = well_tempered_parameter_in;
        threshold_weight        = threshold_weight_in;
        // Resize bin_count and set all entries to 0
        bin_count.assign(edge_number, 0.0);
        // Reset out of range count
        exceeded_count = 0;
        // Print message with parameters
        std::cout << "\nInitialized MetaBiasPotential with the following parameters:\n";
        std::cout << "  CV_min:                  " << CV_min << "\n";
        std::cout << "  CV_max:                  " << CV_max << "\n";
        std::cout << "  bin_number:              " << bin_number << "\n";
        std::cout << "  edge_number:             " << edge_number << "\n";
        std::cout << "  bin_width:               " << bin_width << "\n";
        std::cout << "  bin_width_inverse:       " << bin_width_inverse << "\n";
        std::cout << "  weight:                  " << weight << "\n";
        std::cout << "  well_tempered_parameter: " << well_tempered_parameter << "\n";
        std::cout << "  threshold_weight:        " << threshold_weight << "\n";
        std::cout << "  exceeded_count:          " << exceeded_count << "\n" << std::endl;
    }

    int BinIndexFromCV(const double CV) const noexcept
    {
        return static_cast<int>(std::floor((CV - CV_min) * bin_width_inverse));
    }

    double CVFromBinIndex(const int index) const noexcept
    {
        return CV_min + index * bin_width;
    }

    template<typename funcT>
    void GeneratePotentialFrom(funcT&& generator_function) noexcept
    {
        for (int bin_index = 0; bin_index < edge_number; ++bin_index)
        {
            bin_count[bin_index] = generator_function(CVFromBinIndex(bin_index));
        }
    }

    void UpdatePotential(const double CV) noexcept
    {
        // Get the index to the left of the current CV value
        int bin_index {BinIndexFromCV(CV)};

        // // Linearly interpolating histogram
        // // Casting to unsigned int means we check for values that are in the range [0, edge_number - 1)
        // // Values equal to edge_number - 1 are not allowed when using linear interpolation, since we interpolate between the bin_index and bin_index + 1
        // // Effectively means that our potential is limited to the interval [CV_min, CV_max) when using linear interpolation
        // if (static_cast<unsigned int>(bin_index) < static_cast<unsigned int>(edge_number - 1))
        // {
        //     double interpolation_constant {(CV - CVFromBinIndex(bin_index)) * bin_width_inverse};
        // // Regular
        //     bin_count[bin_index]     += weight * (1.0 - interpolation_constant);
        //     bin_count[bin_index + 1] += weight * interpolation_constant;
        // // Well-tempered
        //     bin_count[bin_index]     += std::exp(-bin_count[bin_index]     / well_tempered_parameter) * weight * (1.0 - interpolation_constant);
        //     bin_count[bin_index + 1] += std::exp(-bin_count[bin_index + 1] / well_tempered_parameter) * weight * interpolation_constant;
        // }

        // Gaussian histogram
        // Casting to unsigned int means we only allow index values that are in the range [0, edge_number)
        if (static_cast<unsigned int>(bin_index) < static_cast<unsigned int>(edge_number))
        {
            for (int bin_index = 0; bin_index < edge_number; ++bin_index)
            {
                double CV_current_bin {CVFromBinIndex(bin_index)};
                if constexpr(metapotential_well_tempered)
                {
                    bin_count[bin_index] += weight * std::exp(-0.5 * bin_width_inverse * bin_width_inverse * std::pow(CV - CV_current_bin, 2) - bin_count[bin_index] / well_tempered_parameter);
                }
                else
                {
                    bin_count[bin_index] += weight * std::exp(-0.5 * bin_width_inverse * bin_width_inverse * std::pow(CV - CV_current_bin, 2));
                }
            }
        }
        else
        {
            exceeded_count += 1;
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

    // Using linear interpolation, return the value of the potential at CV
    double ReturnPotential(const double CV) const noexcept
    {
        // Get the index to the left of the current CV value
        int bin_index {BinIndexFromCV(CV)};

        // Casting to unsigned int means we check for values that are in the range [0, edge_number - 1)
        // Values equal to edge_number - 1 are not allowed, since we interpolate between the bin_index and bin_index + 1
        // Effectively means that our potential is limited to the interval [CV_min, CV_max)
        if (static_cast<unsigned int>(bin_index) < static_cast<unsigned int>(edge_number - 1))
        {
            double interpolation_constant {(CV - CVFromBinIndex(bin_index)) * bin_width_inverse};
            return bin_count[bin_index] * (1.0 - interpolation_constant) + interpolation_constant * bin_count[bin_index + 1];
        }
        else
        {
            // If we fall outside the defined range, return the edge value and an additional quadratic penalty term
            if (bin_index < 0)
            {
                return bin_count.front() + threshold_weight * std::pow((CV - CV_min), 2);
            }
            else
            {
                return bin_count.back()  + threshold_weight * std::pow((CV - CV_max), 2);
            }
        }
    }

    void SymmetrizePotential() noexcept
    {
        // At the time of writing this I'm too lazy to write this without a copy (bin_count_reverse), and this way it at least looks cleaner than manually looping through bin_count
        std::vector<double> bin_count_reverse = bin_count;
        std::reverse_copy(bin_count.cbegin(), bin_count.cend(), bin_count_reverse.begin());
        std::transform(bin_count.begin(), bin_count.end(), bin_count_reverse.cbegin(), bin_count.begin(), [](auto element, auto element_reverse){return 0.5 * (element + element_reverse);});
        std::cout << "Metapotential symmetrized!" << std::endl;
    }

    // Symmetrize by taking the maximum value instead of the average value
    void SymmetrizePotentialMaximum() noexcept
    {
        std::vector<double> bin_count_reverse = bin_count;
        std::reverse_copy(bin_count.cbegin(), bin_count.cend(), bin_count_reverse.begin());
        // TODO: We only need to apply the lambda to the first half of the array (since we always compare the original and reversed array), then we can simply reflect the first half...
        //       Then again, this function is probably called once per run
        std::transform(bin_count.begin(), bin_count.end(), bin_count_reverse.cbegin(), bin_count.begin(), [](auto element, auto element_reverse){return std::max(element, element_reverse);});
        std::cout << "Metapotential symmetrized!" << std::endl;
    }

    // Create a penalty weight for values below CV_lower and values above CV_upper, and write the penalty potential into a file
    [[deprecated]]
    void AddPenaltyWeight(const double CV_lower, const double CV_upper, const std::string& filename) noexcept
    {
        assert(CV_lower < CV_upper);
        std::vector<double> penalty_potential(bin_count.size(), 0.0);
        int lower_index {BinIndexFromCV(CV_lower)};
        int upper_index {BinIndexFromCV(CV_upper)};

        for (int ind = 0; ind < lower_index; ++ind)
        {
            double CV {CVFromBinIndex(ind)};
            penalty_potential[ind] += threshold_weight * std::pow((CV - CV_lower), 2);
        }
        for (int ind = upper_index; ind < edge_number; ++ind)
        {
            double CV {CVFromBinIndex(ind)};
            penalty_potential[ind] += threshold_weight * std::pow((CV - CV_upper), 2);
        }
        std::transform(bin_count.begin(), bin_count.end(), penalty_potential.begin(), bin_count.begin(), std::plus<double>());
        binlog.open(filename, std::fstream::out | std::fstream::app);
        std::copy(penalty_potential.cbegin(), std::prev(penalty_potential.cend()), std::ostream_iterator<double>(binlog, ","));
        binlog << bin_count.back() << "\n";
        binlog.close();
        binlog.clear();
    }

    [[deprecated]]
    void SubtractPenaltyWeight(const double CV_lower, const double CV_upper) noexcept
    {
        assert(CV_lower < CV_upper);
        int lower_index {BinIndexFromCV(CV_lower)};
        int upper_index {BinIndexFromCV(CV_upper)};

        for (int ind = 0; ind < lower_index; ++ind)
        {
            double CV {CVFromBinIndex(ind)};
            bin_count[ind] -= threshold_weight * std::pow((CV - CV_lower), 2);
        }
        for (int ind = upper_index; ind < edge_number; ++ind)
        {
            double CV {CVFromBinIndex(ind)};
            bin_count[ind] -= threshold_weight * std::pow((CV - CV_upper), 2);
        }
    }

    double ReturnBinWidth() const noexcept
    {
        return bin_width;
    }

    double ReturnBinWidthInverse() const noexcept
    {
        return bin_width_inverse;
    }

    double ReturnDerivative(const double CV) const noexcept
    {
        // int bin_index {BinIndexFromCV(CV)};
        // Casting to unsigned int means we check for values that are in the range [0, edge_number - 1)
        // Values equal to edge_number - 1 are not allowed, since we interpolate between the bin_index and bin_index + 1
        // Effectively means that our potential is limited to the interval [CV_min, CV_max)
        // TODO: Shouldn't we calculate the difference between ReturnPotential(bin_index + 1) and bin_index instead of bin_count?
        //       Finally, what about the penalty term in ReturnPotential()? If we directly used ReturnPotential(), this wouldn't be an issue anymore I believe...
        // if (static_cast<unsigned int>(bin_index) < static_cast<unsigned int>(edge_number - 1))
        // {
        //     // double interpolation_constant {(CV - (CV_min + bin_index * bin_width)) * bin_width_inverse};
        //     // std::cout << bin_index << std::endl;
        //     // std::cout << CV << std::endl;
        //     // return 0.5 * (bin_count[bin_index + 1] - bin_count[bin_index]);
        //     return bin_width_inverse * (bin_count[bin_index + 1] - bin_count[bin_index]);
        // }
        // TODO:
        // else
        // {
        //     return;
        // }
        // TODO: Should probably delete the code above and use a symmetric derivative below
        return bin_width_inverse * (ReturnPotential(CV + bin_width) - ReturnPotential(CV));
    }

    void Setweight(const double weight_in) noexcept
    {
        weight = weight_in;
    }

    void SetCV_current(const double CV_in) noexcept
    {
        CV_current = CV_in;
    }

    double ReturnCV_current() const noexcept
    {
        return CV_current;
    }

    void SaveParameters(const std::string& filename, const bool overwrite = false)
    {
        if (overwrite)
        {
            binlog.open(filename, std::fstream::out | std::fstream::trunc);
        }
        else
        {
            binlog.open(filename, std::fstream::out | std::fstream::app);
        }
        binlog << program_version << "\n";
        binlog << "Metadynamics Potential" << "\n";
        binlog << "CV_min: " << CV_min << "\n";
        binlog << "CV_max: " << CV_max << "\n";
        binlog << "bin_number: " << bin_number << "\n";
        binlog << "weight: " << weight << "\n";
        // TODO: Include well_tempered_parameter?
        binlog << "well_tempered_parameter: " << well_tempered_parameter << "\n";
        binlog << "threshold_weight: " << threshold_weight << "\n";
        binlog << "END_METADYN_PARAMS" << "\n";
        binlog.close();
        binlog.clear();
    }

    void SavePotential(const std::string& filename, const bool overwrite = false)
    {
        if (overwrite)
        {
            binlog.open(filename, std::fstream::out | std::fstream::trunc);
        }
        else
        {
            binlog.open(filename, std::fstream::out | std::fstream::app);
        }
        std::copy(bin_count.cbegin(), std::prev(bin_count.cend()), std::ostream_iterator<double>(binlog, ","));
        binlog << bin_count.back() << "\n";
        binlog << "exceeded_count: " << exceeded_count << "\n";
        binlog.close();
        binlog.clear();
    }

    // Function that loads the histogram parameters and the histogram itself from a file
    // TODO: There still seems to be a bug that causes the exceeded count line to be read into the last bin entry?
    // TODO: Work with string_view instead of strings?
    // TODO: This seem to be quite slow: https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring#comment5920160_2602060
    //       Better use stringstreams?
    bool LoadPotential(const std::string& filename)
    {
        if (!std::filesystem::exists(filename))
        {
            std::cerr << Lettuce::Color::BoldRed << "File " << filename << " not found!" << Lettuce::Color::Reset << std::endl;
            return false;
        }
        binload.open(filename, std::fstream::in);
        if (!binload)
        {
            std::cerr << Lettuce::Color::BoldRed << "Reading potential from file " << filename << " failed!" << Lettuce::Color::Reset << std::endl;
        }
        std::string current_line;
        // Count the total linenumber using std::count (add one to the final result since we only counted the number of '\n' characters)
        std::size_t linecount = std::count(std::istreambuf_iterator<char>(binload), std::istreambuf_iterator<char>(), '\n') + 1;
        // Return to beginning of file
        binload.clear();
        binload.seekg(0, binload.beg);
        // Start reading parameters
        // std::getline(binload, current_line);
        // TODO: This seems incorrect...
        // if (current_line == program_version)
        if (true)
        {
            std::cout << "Loading metadynamics potential from: " << filename << "\n";
            std::cout << "The file has " << linecount << " lines in total." << std::endl;
            bool end_token_found {false};
            while(std::getline(binload, current_line))
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
            // Calculate secondary parameters from primary parameters
            bin_width = (CV_max - CV_min) / bin_number;
            bin_width_inverse = 1.0 / bin_width;
            edge_number = bin_number + 1;
            // Warn, but still proceed if 'END_METADYN_PARAMS' is not found
            if (!end_token_found)
            {
                std::cerr << Lettuce::Color::BoldRed << "END_METADYN_PARAMS not found!" << Lettuce::Color::Reset << std::endl;
            }
            // Iterate to the last two lines of the file (linecount - 3, since we still want to read in the second to last line)
            binload.clear();
            binload.seekg(0, binload.beg);
            for (std::size_t ind = 0; ind < linecount - 3; ++ind)
            {
                binload.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
            // Load histogram into bin_count
            bin_count.clear();
            while (std::getline(binload, current_line, ','))
            {
                bin_count.push_back(std::stod(current_line));
            }
            // Get exceeded count
            // TODO: Rewrite in the same fashion as other parameters?
            std::getline(binload, current_line);
            EraseUntil(current_line, "exceeded_count: ", true);
            exceeded_count = std::stoull(current_line);
            // Print message with parameters
            std::cout << "\nSuccessfully loaded MetaBiasPotential from " << filename << " with the following parameters:\n";
            std::cout << "  CV_min:                  " << CV_min << "\n";
            std::cout << "  CV_max:                  " << CV_max << "\n";
            std::cout << "  bin_number:              " << bin_number << "\n";
            std::cout << "  edge_number:             " << edge_number << "\n";
            // TODO: Remove next line after testing
            std::cout << "  bin_count.size():        " << bin_count.size() << "\n";
            std::cout << "  bin_width:               " << bin_width << "\n";
            std::cout << "  bin_width_inverse:       " << bin_width_inverse << "\n";
            std::cout << "  weight:                  " << weight << "\n";
            std::cout << "  well_tempered_parameter: " << well_tempered_parameter << "\n";
            std::cout << "  threshold_weight:        " << threshold_weight << "\n";
            std::cout << "  exceeded_count:          " << exceeded_count << "\n" << std::endl;
            binload.close();
            binload.clear();
            return true;
        }
        else
        {
            std::cout << "Metadynamics potential file comes from an incompatible program version!\n";
            std::cout << "Current version: " << program_version << "\n";
            std::cout << "File version:    " << current_line << "\n";
            binload.close();
            binload.clear();
            return false;
        }
    }
};

#endif // LETTUCE_METADYNAMICS_HPP
