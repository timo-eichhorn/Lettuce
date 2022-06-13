#ifndef LETTUCE_PRNG_WRAPPER_HPP
#define LETTUCE_PRNG_WRAPPER_HPP

// Non-standard library headers
#include "pcg/pcg_random.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <fstream>
#include <string>
#include <vector>
//----------------------------------------
// Standard C headers
// ...

template <typename floatT, typename intT>
class PRNG
{
    public:
        // std::vector<prngT> prng_vec;
        std::vector<pcg64> generator_vec;
        std::vector<std::uniform_real_distribution<floatT>> uniform_real_vec;
        std::vector<std::normal_distribution<floatT>> normal_dist_vec;
        std::vector<std::uniform_int_distribution<intT>> uniform_int_vec;

    PRNG(int n_thread)
    { 
        // std::vector<prngT> prng_vec(n_thread, prngT(seed_source));
        std::vector<pcg64> generator_vec;
        for (int i = 0; i < n_thread; ++i)
        {
            #ifdef FIXED_SEED
            pcg64 generator_temp(i);
            generator_vec.emplace_back(generator_temp);
            // temp_vec.emplace_back(generator_rand_temp(i));
            #else
            pcg_extras::seed_seq_from<std::random_device> seed_source_temp;
            pcg64 generator_rand_temp(seed_source_temp);
            temp_vec.emplace_back(generator_rand_temp);
            // temp_vec.emplace_back(generator_rand_temp(seed_source_temp));
            #endif
        }
        return temp_vec;
        std::vector<std::uniform_real_distribution<floatT>> uniform_real_vec(n_thread, std::uniform_real_distribution<floatT>(0.0, 1.0));
        std::vector<std::normal_distribution<floatT>> normal_dist_vec(n_thread, std::normal_distribution<floatT>(0.0, 1.0));
        std::vector<std::uniform_int_distribution<intT>> uniform_int_vec(n_thread, std::uniform_int_distribution<intT>(1, 8));
    }

    [[nodiscard]]
    floatT UnifReal(int n_thread)
    {
        return uniform_real_vec[n_thread](generator_vec[n_thread]);
    }

    [[nodiscard]]
    floatT NormReal(int n_thread)
    {
        return normal_dist_vec[n_thread](generator_vec[n_thread]);
    }

    [[nodiscard]]
    intT UnifInt(int n_thread)
    {
        return uniform_int_vec[n_thread](generator_vec[n_thread]);
    }

    void SeedPRNG()
    {
        for (auto& element : generator_vec)
        {
            pcg_extras::seed_seq_from<std::random_device> seed_source_temp;
            pcg64 generator_rand_temp(seed_source_temp);
            element = generator_rand_temp;
        }   
    }

    // TODO: Need to save the distribution states as well (at least for the normal distribution)
    void SaveState(const std::string& filename, const bool overwrite = false)
    {
        std::ofstream state_stream_out(filename, std::ios::binary);
        for (auto& prng_el : prng_vec)
        {
            state_stream_out << prng_el;
            // TODO: Some kind of delimiter??
        }
    }

    // TODO: Need to save the distribution states as well (at least for the normal distribution)
    void LoadState(const std::string& filename)
    {
        std::ifstream state_stream_in(filename, std::ios::binary);
        // TODO: size assertion or something like that?
        // prng_vec.resize();
        for (auto& prng_el : prng_vec)
        {
            state_stream_in >> prng_el;
            // TODO: Some kind of delimiter??
        }

    }
    private:
        // vector<prngT> prng_vec;
};

#endif // LETTUCE_PRNG_WRAPPER_HPP
