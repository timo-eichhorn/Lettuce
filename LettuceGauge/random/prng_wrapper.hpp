#ifndef LETTUCE_PRNG_WRAPPER_HPP
#define LETTUCE_PRNG_WRAPPER_HPP

// Non-standard library headers
#include "../coords.hpp"
#include "../IO/ansi_colors.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <concepts>
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

// TODO: Use unsigned long long or more generic integral type?
// Note that this doesn't give any complexity bounds on the discard function
template<typename T>
concept SkippablePRNG = requires(T prng, unsigned long long skip_distance)
{
    {prng.discard(skip_distance)} -> std::same_as<void>;
};

// Per default, use floatT set in defines.hpp as floating point type and standard ints as integer type
// template<typename prngT, typename floatT = floatT, typename intT = int>
template<int Nt_, int Nx_, int Ny_, int Nz_, SkippablePRNG prngT, typename floatT, typename intT = int>
class PRNG4D
{
    private:
        static constexpr int         Nt   {Nt_};
        static constexpr int         Nx   {Nx_};
        static constexpr int         Ny   {Ny_};
        static constexpr int         Nz   {Nz_};
        static constexpr int         Nmu  {4};
        // Promote single length to size_t so the product doesn't overflow
        static constexpr std::size_t size {static_cast<std::size_t>(Nt) * Nx * Ny * Nz * Nmu};

        // This is copied straight from pcg_extras (comes with the PCG PRNG by Melissa O'Neill: http://www.pcg-random.org/)
        template<typename RngType>
        class seed_seq_from
        {
            private:
                RngType rng_;
            public:
                // typedef uint_least32_t result_type;
                using result_type = uint_least32_t;

                template<typename... Args>
                seed_seq_from(Args&&... args) :
                rng_(std::forward<Args>(args)...)
                {}

                template<typename Iter>
                void generate(Iter start, Iter finish)
                {
                    for (auto i = start; i != finish; ++i)
                    {
                        *i = result_type(rng_());
                    }
                }

                constexpr size_t size() const
                {
                    return (sizeof(typename RngType::result_type) > sizeof(result_type)
                            && RngType::max() > ~size_t(0UL))
                         ? ~size_t(0UL)
                         : size_t(RngType::max());
                }
        };
    public:
        std::vector<prngT>                                  random_generators;
        // uniform_real_distribution and uniform_int_distribution should be thread-safe, so we might only need one instance of each
        // Unfortunately normal_distribution is not thread-safe since it has internal state, so to be able to correcly use it in parallel we need multiple (n_thread_max) instances
        // It is probably easiest to simply generate one instance of normal_distribution per lattice link
        std::vector<std::normal_distribution<floatT>>       normal_distributions;
        std::vector<std::uniform_real_distribution<floatT>> uniform_real_distributions;
        std::vector<std::uniform_int_distribution<intT>>    uniform_int_distributions;
        using prng_type       = prngT;
        // TODO: Alias for state type and size of prngT like in Grid? Problematic, since not all PRNGs seem to provide a way to check the state size?
        using float_type      = floatT;
        using int_type        = intT;

        template<typename seed_sourceT>
        explicit PRNG4D(seed_sourceT&& seed_source) noexcept :
        random_generators(size), normal_distributions(size, std::normal_distribution<floatT>(0.0, 1.0)), uniform_real_distributions(size, std::uniform_real_distribution<floatT>(0.0, 1.0)), uniform_int_distributions(size, std::uniform_int_distribution<intT>(1, 8))
        {
            #ifdef FIXED_SEED
            // Ignore seed source and seed all PRNGs incrementally (seeding all PRNGs with the same seed will cause the HMC to not work properly, since all initial momenta are the same)
            for (std::size_t index = 0; index < size; ++index)
            {
                SeedPRNG(index, index);
            }
            #else
            // Seed all PRNGs with the provided seed_source
            SeedPRNGs(std::forward<seed_sourceT>(seed_source));
            #endif
        }

        // Shuffle
        // void ShuffleStates() noexcept
        // {
        //     //
        // }

        // (Re)seed all PRNGs with LCG
        // void SeedPRNGs() noexcept
        // {
        //     for (auto& prng : random_generators)
        //     {
        //         prng.seed();
        //     }
        // }

        // void SkipAhead(unsigned long long skip_distance) noexcept
        // {
        //     for (std::size_t index = 0; index < size; ++index)
        //     {
        //         unsigned long long total_skip_distance {index * skip_distance};
        //         random_generators[index].skip(total_skip_distance);
        //     }
        // }

        // (Re)seed all PRNGs with a custom seed source
        template<typename seed_sourceT>
        void SeedPRNGs(seed_sourceT&& seed_source) noexcept
        {
            seed_seq_from<seed_sourceT> seed_seq(seed_source);
            // For now seed the PRNGs sequentially; not yet sure if we can parallelize this without knowing anything about seed_source
            for (auto& prng : random_generators)
            {
                prng.seed(seed_seq);
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
        void SavePRNGState(const std::string& filename, const bool overwrite = false)
        {
            if (std::filesystem::exists(filename))
            {
                std::cout << Lettuce::Color::BoldRed << "File " << filename << " already exists!" << Lettuce::Color::Reset << std::endl;
                if (overwrite)
                {
                    std::cout << Lettuce::Color::BoldRed << "Overwriting existing file..." << Lettuce::Color::Reset << std::endl;
                }
                else
                {
                    std::cerr << Lettuce::Color::BoldRed << "Writing PRNG state to file " << filename << " failed!" << Lettuce::Color::Reset << std::endl;
                    return;
                }
            }
            std::ofstream state_stream(filename, std::ios::trunc);
            state_stream.imbue(std::locale("C"));
            for (auto& prng : random_generators)
            {
                state_stream << prng << " ";
            }
            state_stream.close();
            //-----
            if (!state_stream)
            {
                std::cerr << Lettuce::Color::BoldRed << "Writing PRNG state to file " << filename << " failed!" << Lettuce::Color::Reset << std::endl;
            }
        }

        // lttc_host
        void LoadPRNGState(const std::string& filename)
        {
            if (!std::filesystem::exists(filename))
            {
                std::cerr << Lettuce::Color::BoldRed << "Error while trying to load PRNG state: File " << filename << " not found!" << Lettuce::Color::Reset << std::endl;
                return;
            }
            //-----
            std::ifstream state_stream(filename);
            state_stream.imbue(std::locale("C"));
            for (auto& prng : random_generators)
            {
                state_stream >> prng;
            }
            //-----
            if (!state_stream)
            {
                std::cerr << Lettuce::Color::BoldRed << "Reading PRNG state from file " << filename << " failed!" << Lettuce::Color::Reset << std::endl;
            }
            state_stream.close();
        }

        // lttc_host
        void SaveDistributionState(const std::string& filename, const bool overwrite = false)
        {
            if (std::filesystem::exists(filename))
            {
                std::cout << Lettuce::Color::BoldRed << "File " << filename << " already exists!" << Lettuce::Color::Reset << std::endl;
                if (overwrite)
                {
                    std::cout << Lettuce::Color::BoldRed << "Overwriting existing file..." << Lettuce::Color::Reset << std::endl;
                }
                else
                {
                    std::cerr << Lettuce::Color::BoldRed << "Writing normal distribution state to file " << filename << " failed!" << Lettuce::Color::Reset << std::endl;
                    return;
                }
            }
            std::ofstream state_stream(filename, std::ios::trunc);
            state_stream.imbue(std::locale("C"));
            for (auto& normal_distribution : normal_distributions)
            {
                state_stream << normal_distribution << " ";
            }
            state_stream.close();
            //-----
            if (!state_stream)
            {
                std::cerr << Lettuce::Color::BoldRed << "Writing normal distribution state to file " << filename << " failed!" << Lettuce::Color::Reset << std::endl;
            }
        }

        // lttc_host
        void LoadDistributionState(const std::string& filename)
        {
            if (!std::filesystem::exists(filename))
            {
                std::cerr << Lettuce::Color::BoldRed << "Error while trying to load normal distribution state: File " << filename << " not found!" << Lettuce::Color::Reset << std::endl;
                return;
            }
            //-----
            std::ifstream state_stream(filename);
            state_stream.imbue(std::locale("C"));
            for (auto& normal_distribution : normal_distributions)
            {
                state_stream >> normal_distribution;
            }
            //-----
            if (!state_stream)
            {
                std::cerr << Lettuce::Color::BoldRed << "Reading normal distribution state from file " << filename << " failed!" << Lettuce::Color::Reset << std::endl;
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
        void SaveState(const std::string& filename_prng, const std::string& filename_normal_distribution, const bool overwrite = false)
        {
            SavePRNGState(filename_prng, overwrite);
            SaveDistributionState(filename_normal_distribution, overwrite);
        }

        [[nodiscard]]
        floatT UniformReal(const std::size_t index = 0) noexcept
        {
            return uniform_real_distributions[index](random_generators[index]);
        }

        [[nodiscard]]
        floatT UniformReal(const site_coord& site, const int mu) noexcept
        {
            std::size_t index {LinearCoordinate(site, mu)};
            return uniform_real_distributions[index](random_generators[index]);
        }

        [[nodiscard]]
        floatT UniformReal(const link_coord& link) noexcept
        {
            std::size_t index {LinearCoordinate(link)};
            return uniform_real_distributions[index](random_generators[index]);
        }

        [[nodiscard]]
        floatT UniformReal(const int t, const int x, const int y, const int z, const int mu) noexcept
        {
            std::size_t index {LinearCoordinate({t, x, y, z, mu})};
            return uniform_real_distributions[index](random_generators[index]);
        }
        //-----
        [[nodiscard]]
        floatT Gaussian(const std::size_t index = 0) noexcept
        {
            return normal_distributions[index](random_generators[index]);
        }

        [[nodiscard]]
        floatT Gaussian(const site_coord& site, const int mu) noexcept
        {
            std::size_t index {LinearCoordinate(site, mu)};
            return normal_distributions[index](random_generators[index]);
        }

        [[nodiscard]]
        floatT Gaussian(const link_coord& link) noexcept
        {
            std::size_t index {LinearCoordinate(link)};
            return normal_distributions[index](random_generators[index]);
        }

        [[nodiscard]]
        floatT Gaussian(const int t, const int x, const int y, const int z, const int mu) noexcept
        {
            std::size_t index {LinearCoordinate({t, x, y, z, mu})};
            return normal_distributions[index](random_generators[index]);
        }
        //-----
        [[nodiscard]]
        intT UniformInt(const std::size_t index = 0) noexcept
        {
            return uniform_int_distributions[index](random_generators[index]);
        }

        [[nodiscard]]
        intT UniformInt(const site_coord& site, const int mu) noexcept
        {
            std::size_t index {LinearCoordinate(site, mu)};
            return uniform_int_distributions[index](random_generators[index]);
        }

        [[nodiscard]]
        intT UniformInt(const link_coord& link) noexcept
        {
            std::size_t index {LinearCoordinate(link)};
            return uniform_int_distributions[index](random_generators[index]);
        }

        [[nodiscard]]
        intT UniformInt(const int t, const int x, const int y, const int z, const int mu) noexcept
        {
            std::size_t index {LinearCoordinate({t, x, y, z, mu})};
            return uniform_int_distributions[index](random_generators[index]);
        }
    private:
        //-----
        // TODO: In the future, include Nt, Nx, Ny, Nz, Nmu via some kind of underlying lattice geometry object
        [[nodiscard]]
        inline std::size_t LinearCoordinate(const site_coord& site, const int mu) const noexcept
        {
            return (((site.t * static_cast<std::size_t>(Nx) + site.x) * Ny + site.y) * Nz + site.z) * Nmu + mu;
        }
        [[nodiscard]]
        inline std::size_t LinearCoordinate(const link_coord& coord) const noexcept
        {
            return (((coord.t * static_cast<std::size_t>(Nx) + coord.x) * Ny + coord.y) * Nz + coord.z) * Nmu + coord.mu.direction;
        }
        [[nodiscard]]
        inline std::size_t LinearCoordinate(const int t, const int x, const int y, const int z, const int mu) const noexcept
        {
            return (((t * static_cast<std::size_t>(Nx) + x) * Ny + y) * Nz + z) * Nmu + mu;
        }
};

#endif // LETTUCE_PRNG_WRAPPER_HPP
