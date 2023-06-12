#ifndef LETTUCE_METADYNAMICS_HPP
#define LETTUCE_METADYNAMICS_HPP

// Non-standard library headers
#include "IO/ansi_colors.hpp"
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
#include <string>
#include <utility>
#include <vector>
//----------------------------------------
// Standard C headers
#include <cassert>
#include <cmath>

class MetaBiasPotential
{
private:
    std::vector<double> bin_count;
    double CV_current;
    double CV_min, CV_max;
    size_t bin_number;
    size_t edge_number;
    double bin_width, bin_width_inverse, weight, threshold_weight;
    uint_fast64_t exceeded_count;
    std::ofstream binlog;
    std::ifstream binload;
public:
    // TODO: Rewrite constructor
    // MetaBiasPotential(const double CV_min_in, const double CV_max_in, const size_t bin_number_in, const double weight_in, const double threshold_weight_in) noexcept :
    // CV_min(CV_min_in), CV_max(CV_max_in), bin_number(bin_number_in), edge_number(bin_number_in + 1) weight(weight_in), threshold_weight(threshold_weight_in)
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
    MetaBiasPotential(const double CV_min_in, const double CV_max_in, const size_t bin_number_in, const double weight_in, const double threshold_weight_in)
    {
        assert(CV_min_in < CV_max_in);
        CV_min = CV_min_in;
        CV_max = CV_max_in;
        bin_number = bin_number_in;
        edge_number = bin_number + 1;
        bin_width = (CV_max - CV_min) / bin_number;
        bin_width_inverse = 1.0 / bin_width;
        weight = weight_in;
        threshold_weight = threshold_weight_in;
        // Resize bin_count and set all entries to 0
        bin_count.assign(edge_number, 0.0);
        // Reset out of range count
        exceeded_count = 0;
        // Print message with parameters
        std::cout << "\nInitialized MetaBiasPotential with the following parameters:\n";
        std::cout << "  CV_min: " << CV_min << "\n";
        std::cout << "  CV_max: " << CV_max << "\n";
        std::cout << "  bin_number: " << bin_number << "\n";
        std::cout << "  edge_number: " << edge_number << "\n";
        std::cout << "  bin_width: " << bin_width << "\n";
        std::cout << "  bin_width_inverse: " << bin_width_inverse << "\n";
        std::cout << "  weight: " << weight << "\n";
        std::cout << "  threshold_weight: " << threshold_weight << "\n";
        std::cout << "  exceeded_count: " << exceeded_count << "\n" << std::endl;
    }

    //-----
    // Function that updates the histogram
    // TODO: Add additional weight parameter to accelerate the creation of the potential for topological updates?

    void UpdatePotential(const double CV) noexcept
    {
        // TODO: floor or round?
        int bin_index {static_cast<int>(std::floor((CV - CV_min) * bin_width_inverse))};
        // Effectively means that our potential is limited to the interval [CV_min, CV_max)
        // if ((unsigned int)(bin_index) <= (unsigned int)(bin_number - 1))
        // if ((unsigned int)(bin_index) < (unsigned int)(bin_number - 1))
        // if ((unsigned int)(bin_index) < (unsigned int)(edge_number - 1))

        if (static_cast<unsigned int>(bin_index) < static_cast<unsigned int>(edge_number - 1))
        {
            // double interpolation_constant {(CV - (CV_min + bin_index * bin_width)) * bin_width_inverse};
            // bin_count[bin_index] += weight * (1.0 - interpolation_constant);
            // bin_count[bin_index + 1] += weight * interpolation_constant;
            // Gaussian histogram?
            for (int bin_index = 0; bin_index < edge_number; ++bin_index)
            {
                double CV_current_bin {CV_min + bin_index * bin_width};
                bin_count[bin_index] += weight * std::exp(-0.5 * bin_width_inverse * bin_width_inverse * std::pow(CV - CV_current_bin, 2));
            }
        }
        else
        {
            exceeded_count += 1;
        }
    }

    // void UpdatePotentialWellTempered(const double CV) noexcept
    // {
    //     int bin_index {static_cast<int>(std::floor((CV - CV_min) * bin_width_inverse))};
    //     // Effectively means that our potential is limited to the interval [CV_min, CV_max)

    //     if (static_cast<unsigned int>(bin_index) < static_cast<unsigned int>(edge_number - 1))
    //     {
    //         // Gaussian histogram?
    //         // In the well tempered variant, we have an additional exponential factor in front that depends on the value of the current bin
    //         for (int bin_index = 0; bin_index < edge_number; ++bin_index)
    //         {
    //             double CV_current_bin {CV_min + bin_index * bin_width};
    //             bin_count[bin_index] += std::exp(-bin_count[bin_index] * ) * weight * std::exp(-0.5 * bin_width_inverse * bin_width_inverse * std::pow(CV - CV_current_bin, 2));
    //         }
    //     }
    //     else
    //     {
    //         exceeded_count += 1;
    //     }
    // }

    //-----
    // Function that returns the histogram entry

    double ReturnPotential(const double CV) const noexcept
    {
        int bin_index {static_cast<int>(std::floor((CV - CV_min) * bin_width_inverse))};
        // Effectively means that our potential is limited to the interval [CV_min, CV_max)
        // if ((unsigned int)(bin_index) <= (unsigned int)(bin_number - 1))
        // if ((unsigned int)(bin_index) < (unsigned int)(bin_number - 1))
        // if ((unsigned int)(bin_index) < (unsigned int)(edge_number - 1))
        if (static_cast<unsigned int>(bin_index) < static_cast<unsigned int>(edge_number - 1))
        {
            double interpolation_constant {(CV - (CV_min + bin_index * bin_width)) * bin_width_inverse};
            return bin_count[bin_index] * (1.0 - interpolation_constant) + interpolation_constant * bin_count[bin_index + 1];
        }
        else
        {
            // Original version
            // return threshold_weight * std::min(std::pow((CV - CV_min), 2), std::pow((CV - CV_max), 2));
            // Second version
            // Having both differences is redundant for symmetric potentials, but this way it works for arbitrary potentials
            // return threshold_weight * std::min(std::abs(CV * CV - CV_min * CV_min), std::abs(CV * CV - CV_max * CV_max));
            // Third, most extreme version
            return threshold_weight * (0.1 + std::min(std::pow((CV - CV_min), 2), std::pow((CV - CV_max), 2)));
        }
    }

    //-----
    // Symmetrize potential
    void SymmetrizePotential() noexcept
    {
        // At the time of writing this I'm too lazy to write this without a copy (bin_count_reverse), and this way it at least looks cleaner than manually looping through bin_count
        // TODO: Ranges to the rescue to have a reverse view?
        std::vector<double> bin_count_reverse = bin_count;
        std::reverse_copy(bin_count.cbegin(), bin_count.cend(), bin_count_reverse.begin());
        std::transform(bin_count.begin(), bin_count.end(), bin_count_reverse.cbegin(), bin_count.begin(), [](auto element, auto element_reverse){return 0.5 * (element + element_reverse);});
        std::cout << "Metapotential symmetrized!" << std::endl;
    }

    // Symmetrize, but do not take the average value, but rather the maximum value
    void SymmetrizePotentialMaximum() noexcept
    {
        // TODO: We only need to check the maximum for the first half of the array, then we can simply reflect the first half...
        //       Then again, this function is probably called once per run
        std::vector<double> bin_count_reverse = bin_count;
        std::reverse_copy(bin_count.cbegin(), bin_count.cend(), bin_count_reverse.begin());
        std::transform(bin_count.begin(), bin_count.end(), bin_count_reverse.cbegin(), bin_count.begin(), [](auto element, auto element_reverse){return std::max(element, element_reverse);});
        std::cout << "Metapotential symmetrized!" << std::endl;
    }

    //-----
    // Create a penalty weight starting for values below CV_lower and values above CV_upper

    void AddPenaltyWeight(const double CV_lower, const double CV_upper, const std::string& filename) noexcept
    {
        assert(CV_lower < CV_upper);
        int lower_index {static_cast<int>(std::floor((CV_lower - CV_min) * bin_width_inverse))};
        int upper_index {static_cast<int>(std::floor((CV_upper - CV_min) * bin_width_inverse))};
        // for (int ind = 0; ind < lower_index; ++ind)
        // {
        //     double CV = CV_min + ind * bin_width;
        //     bin_count[ind] += threshold_weight * std::pow((CV - CV_lower), 2);
        // }
        // for (int ind = upper_index; ind < edge_number; ++ind)
        // {
        //     double CV = CV_min + ind * bin_width;
        //     bin_count[ind] += threshold_weight * std::pow((CV - CV_upper), 2);
        // }
        std::vector<double> penalty_potential(bin_count.size(), 0.0);
        for (int ind = 0; ind < lower_index; ++ind)
        {
            double CV {CV_min + ind * bin_width};
            penalty_potential[ind] += threshold_weight * std::pow((CV - CV_lower), 2);
        }
        for (int ind = upper_index; ind < edge_number; ++ind)
        {
            double CV {CV_min + ind * bin_width};
            penalty_potential[ind] += threshold_weight * std::pow((CV - CV_upper), 2);
        }
        std::transform(bin_count.begin(), bin_count.end(), penalty_potential.begin(), bin_count.begin(), std::plus<double>());
        binlog.open(filename, std::fstream::out | std::fstream::app);
        std::copy(penalty_potential.cbegin(), std::prev(penalty_potential.cend()), std::ostream_iterator<double>(binlog, ","));
        binlog << bin_count.back() << "\n";
        binlog.close();
        binlog.clear();
    }

    void SubtractPenaltyWeight(const double CV_lower, const double CV_upper) noexcept
    {
        assert(CV_lower < CV_upper);
        int lower_index {static_cast<int>(std::floor((CV_lower - CV_min) * bin_width_inverse))};
        int upper_index {static_cast<int>(std::floor((CV_upper - CV_min) * bin_width_inverse))};
        for (int ind = 0; ind < lower_index; ++ind)
        {
            double CV {CV_min + ind * bin_width};
            bin_count[ind] -= threshold_weight * std::pow((CV - CV_lower), 2);
        }
        for (int ind = upper_index; ind < edge_number; ++ind)
        {
            double CV {CV_min + ind * bin_width};
            bin_count[ind] -= threshold_weight * std::pow((CV - CV_upper), 2);
        }
    }

    //-----
    // Returns bin_width

    double ReturnBinWidth() const noexcept
    {
        return bin_width;
    }

    //-----
    // Returns bin_width_inverse (required for HMC force calculation)

    double ReturnBinWidthInverse() const noexcept
    {
        return bin_width_inverse;
    }

    double ReturnDerivative(const double CV) const noexcept
    {
        // TODO: Add bounds checks just like with ReturnPotential()?
        //       Also what about the penalty potential?
        int bin_index {static_cast<int>(std::floor((CV - CV_min) * bin_width_inverse))};
        // double interpolation_constant {(CV - (CV_min + bin_index * bin_width)) * bin_width_inverse};
        // std::cout << bin_index << std::endl;
        // std::cout << CV << std::endl;
        // return 0.5 * (bin_count[bin_index + 1] - bin_count[bin_index]);
        return bin_width_inverse * (bin_count[bin_index + 1] - bin_count[bin_index]);
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

    //-----
    // Function that saves the parameters to a file

    void SaveMetaParameters(const std::string& filename, const bool overwrite = false)
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
        // TODO: Write penalty potential parameters to file?
        // binlog << "CV_pen_min: " << CV_pen_min << "\n";
        // binlog << "CV_pen_max: " << CV_pen_max << "\n";
        binlog << "bin_number: " << bin_number << "\n";
        binlog << "weight: " << weight << "\n";
        binlog << "threshold_weight: " << threshold_weight << "\n";
        binlog << "END_METADYN_PARAMS" << "\n";
        binlog.close();
        binlog.clear();
    }

    //-----
    // Function that saves the histogram to a file

    void SaveMetaPotential(const std::string& filename, const bool overwrite = false)
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

    //-----
    // C++ strings suck, so define this helper function
    // TODO: Eventually remove this function from this struct and move into utility library

    void LeftErase(std::string& str, const std::string& erase)
    {
        size_t pos = str.find(erase);
        if (pos != std::string::npos)
        {
            str.erase(pos, erase.length());
        }
    }

    //-----
    // Search the string 'str' for 'erase' starting and delete everything to the left of 'erase'
    // If including == true, erase everything including 'erase', otherwise only until 'erase'

    void EraseUntil(std::string& str, const std::string& erase, const bool including = true)
    {
        size_t pos = str.find(erase);
        if (pos != std::string::npos)
        {
            if (including)
            {
                str.erase(0, pos + erase.length());
            }
            else
            {
                str.erase(0, pos);
            }
        }
    }

    //-----
    // Function that loads the histogram parameters and the histogram itself from a file
    // TODO: There still seems to be a bug that causes the exceeded count line to be read into the last bin entry?
    // TODO: Work with string_view instead of strings?
    // TODO: This seem to be quite slow: https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring#comment5920160_2602060
    //       Better use stringstreams?

    bool LoadPotential(const std::string& filename)
    {
        if (!std::filesystem::exists(filename))
        {
            std::cout << Lettuce::Color::BoldRed << "File " << filename << " not found!" << Lettuce::Color::Reset << std::endl;
            return false;
        }
        binload.open(filename, std::fstream::in);
        std::string current_line;
        // Count the total linenumber using std::count (add one to the final result since we only counted the number of '\n' characters)
        size_t linecount = std::count(std::istreambuf_iterator<char>(binload), std::istreambuf_iterator<char>(), '\n') + 1;
        // Return to beginning of file
        binload.clear();
        binload.seekg(0, binload.beg);
        // Start reading parameters
        std::getline(binload, current_line);
        // TODO: This seems incorrect...
        // if (current_line == program_version)
        if (true)
        {
            std::cout << "Loading metadynamics potential from: " << filename << "\n";
            std::cout << "The file has " << linecount << " lines in total." << std::endl;
            // Skip second line
            std::getline(binload, current_line);
            // Get CV_min
            std::getline(binload, current_line);
            LeftErase(current_line, "CV_min: ");
            CV_min = std::stod(current_line);
            // Get CV_max
            std::getline(binload, current_line);
            LeftErase(current_line, "CV_max: ");
            CV_max = std::stod(current_line);
            // TODO: Read penalty potential parameters from file?
            // Get CV_pen_min
            // std::getline(binload, current_line);
            // LeftErase(current_line, "CV_pen_min: ");
            // CV_pen_min = std::stod(current_line);
            // Get CV_pen_max
            // std::getline(binload, current_line);
            // LeftErase(current_line, "CV_pen_max: ");
            // CV_pen_max = std::stod(current_line);
            // Get bin_number
            std::getline(binload, current_line);
            LeftErase(current_line, "bin_number: ");
            bin_number = std::stoi(current_line);
            // Get weight
            std::getline(binload, current_line);
            LeftErase(current_line, "weight: ");
            weight = std::stod(current_line);
            // Get threshold_weight
            std::getline(binload, current_line);
            LeftErase(current_line, "threshold_weight: ");
            threshold_weight = std::stod(current_line);
            // Iterate to the last two lines of the file (linecount - 3, since we still want to read in the second to last line)
            binload.clear();
            binload.seekg(0, binload.beg);
            for (size_t ind = 0; ind < linecount - 3; ++ind)
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
            std::getline(binload, current_line);
            EraseUntil(current_line, "exceeded_count: ", true);
            exceeded_count = std::stoull(current_line);
            // Calculate secondary parameters from primary parameters
            // TODO: Use Init function?
            bin_width = (CV_max - CV_min) / bin_number;
            bin_width_inverse = 1.0 / bin_width;
            edge_number = bin_number + 1;
            // Print message with parameters
            std::cout << "\nSuccessfully loaded MetaBiasPotential from " << filename << " with the following parameters:\n";
            std::cout << "  CV_min: " << CV_min << "\n";
            std::cout << "  CV_max: " << CV_max << "\n";
            std::cout << "  bin_number: " << bin_number << "\n";
            std::cout << "  edge_number: " << edge_number << "\n";
            // TODO: Remove next line after testing
            std::cout << "  bin_count.size(): " << bin_count.size() << "\n";
            std::cout << "  bin_width: " << bin_width << "\n";
            std::cout << "  bin_width_inverse: " << bin_width_inverse << "\n";
            std::cout << "  weight: " << weight << "\n";
            std::cout << "  threshold_weight: " << threshold_weight << "\n";
            std::cout << "  exceeded_count: " << exceeded_count << "\n" << std::endl;
            binload.close();
            binload.clear();
            return true;
        }
        else
        {
            std::cout << "Metadynamics potential file comes from an incompatible program version!\n";
            std::cout << "Current version: " << program_version << "\n";
            std::cout << "File version: " << current_line << "\n";
            binload.close();
            binload.clear();
            return false;
        }
    }
};

#endif // LETTUCE_METADYNAMICS_HPP
