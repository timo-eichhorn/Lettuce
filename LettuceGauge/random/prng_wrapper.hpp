#ifndef LETTUCE_PRNG_WRAPPER_HPP
#define LETTUCE_PRNG_WRAPPER_HPP

// Non-standard library headers
#include "../IO/ansi_colors.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>
//----------------------------------------
// Standard C headers
#include <cstddef>

//+---------------------------------------------------------------------------------+
//| This file provides a PRNG wrapper class that holds not only the PRNGs, but also |
//| some of the random distributions commonly required for lattice simulations:     |
//| - Normal/Gaussian distribution (used during HMC)                                |
//| - Real distribution on [0, 1) (mostly used for probabilites)                    |
//| - Uniform int distribution on [1, 8] (used to get random su(3) direction)       |
//| Additionally, some utility functions are provided that allow the (re)seeding of |
//| the PRNGs and saving/loading the states to/from files.                          |
//+---------------------------------------------------------------------------------+

// Per default, use floatT set in defines.hpp as floating point type and standard ints as integer type
// template<typename prngT, typename floatT = floatT, typename intT = int>
template<typename prngT, typename floatT, typename intT = int>
class PRNG
{
    private:
        // ...
    public:
        std::vector<prngT>                                  random_generators;
        // uniform_real_distribution and uniform_int_distribution should be thread-safe, so we might only need one instance of each
        // Unfortunately normal_distribution is not thread-safe since it has internal state, so to be able to correcly use it in parallel we need multiple (n_thread_max) instances
        // It is probably easiest to simply generate one instance of normal_distribution per lattice site (not link!)
        std::vector<std::normal_distribution<floatT>>       normal_distributions;
        std::vector<std::uniform_real_distribution<floatT>> uniform_real_distributions;
        std::vector<std::uniform_int_distribution<intT>>    uniform_int_distributions;
        std::size_t                                         n_prng;
        using prng_type       = prngT;
        // TODO: Alias for state type and size of prngT like in Grid? Problematic, since not all PRNGs seem to provide a way to check the state size?
        using float_type      = floatT;
        using int_type        = intT;

        template<typename seed_sourceT>
        explicit PRNG(const std::size_t n_prng_in, seed_sourceT&& seed_source) noexcept :
        random_generators(n_prng_in), normal_distributions(n_prng_in, std::normal_distribution<floatT>(0.0, 1.0)), uniform_real_distributions(n_prng_in, std::uniform_real_distribution<floatT>(0.0, 1.0)), uniform_int_distributions(n_prng_in, std::uniform_int_distribution<intT>(1, 8)), n_prng(n_prng_in)
        {
            #ifdef FIXED_SEED
            // Ignore seed_source and seed all PRNGs with 1
            SeedPRNGs(1);
            #else
            // Seed all PRNGs with the provided seed_source
            SeedPRNGs(std::forward<seed_sourceT>(seed_source));
            #endif
        }

        // (Re)seed all PRNGs
        template<typename seed_sourceT>
        void SeedPRNGs(seed_sourceT&& seed_source) noexcept
        {
            // For now seed the PRNGs sequentially; not yet sure if we can parallelize this without knowing anything about seed_source
            for (auto& prng : random_generators)
            {
                prng.seed(seed_source);
            }
        }

        // (Re)seed single PRNG
        template<typename seed_sourceT>
        void SeedPRNG(const std::size_t prng_index, seed_sourceT&& seed_source) noexcept
        {
            random_generators[prng_index].seed(seed_source);
        }

        // TODO: Is there a portable way to write the states in binary format?
        // TODO: When using CUDA, we can't write to files from device functions, so (only?) provide functions that save/load state to/from vector?
        // lttc_host
        void SavePRNGState(const std::string& filename)
        {
            std::ofstream state_stream(filename);
            for (auto& prng : random_generators)
            {
                state_stream << prng << " ";
            }
            state_stream.close();
            //-----
            if (!state_stream)
            {
                std::cerr << Lettuce::Color::BoldRed << "Writing PRNG state to file " << filename << "failed!" << Lettuce::Color::Reset << std::endl;
            }
        }

        // lttc_host
        void LoadPRNGState(const std::string& filename)
        {
            if (!std::filesystem::exists(filename))
            {
                std::cerr << Lettuce::Color::BoldRed << "Error while trying to load PRNG state: File " << filename << "not found!" << Lettuce::Color::Reset << std::endl;
                return;
            }
            //-----
            std::ifstream state_stream(filename);
            for(auto& prng : random_generators)
            {
                state_stream >> prng;
            }
            //-----
            if (!state_stream)
            {
                std::cerr << Lettuce::Color::BoldRed << "Reading PRNG state from file " << filename << "failed!" << Lettuce::Color::Reset << std::endl;
            }
            state_stream.close();
        }

        // lttc_host
        void SaveDistributionState(const std::string& filename)
        {
            std::ofstream state_stream(filename);
            for (auto& normal_distribution : normal_distributions)
            {
                state_stream << normal_distribution << " ";
            }
            state_stream.close();
            //-----
            if (!state_stream)
            {
                std::cerr << Lettuce::Color::BoldRed << "Writing PRNG state to file " << filename << "failed!" << Lettuce::Color::Reset << std::endl;
            }
        }

        // lttc_host
        void LoadDistributionState(const std::string& filename)
        {
            if (!std::filesystem::exists(filename))
            {
                std::cerr << Lettuce::Color::BoldRed << "Error while trying to load normal distribution state: File " << filename << "not found!" << Lettuce::Color::Reset << std::endl;
                return;
            }
            //-----
            std::ifstream state_stream(filename);
            for(auto& normal_distribution : normal_distributions)
            {
                state_stream >> normal_distribution;
            }
            //-----
            if (!state_stream)
            {
                std::cerr << Lettuce::Color::BoldRed << "Reading normal distribution state from file " << filename << "failed!" << Lettuce::Color::Reset << std::endl;
            }
            state_stream.close();
        }

        // lttc_host
        void LoadState(const std::string& filename_prng, const std::string& filename_normal_distribution)
        {
            LoadPRNGState(filename_prng);
            LoadDistributionState(filename_normal_distribution);
        }

        // lttc_host
        void SaveState(const std::string& filename_prng, const std::string& filename_normal_distribution)
        {
            SavePRNGState(filename_prng);
            SaveDistributionState(filename_normal_distribution);
        }

        [[nodiscard]]
        floatT UniformReal(const std::size_t index) noexcept
        {
            return uniform_real_distributions[index](random_generators[index]);
        }

        [[nodiscard]]
        floatT Gaussian(const std::size_t index) noexcept
        {
            return normal_distributions[index](random_generators[index]);
        }

        [[nodiscard]]
        intT UniformInt(const std::size_t index) noexcept
        {
            return uniform_int_distributions[index](random_generators[index]);
        }
};

#endif // LETTUCE_PRNG_WRAPPER_HPP
