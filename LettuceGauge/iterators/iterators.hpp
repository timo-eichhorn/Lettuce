#ifndef LETTUCE_ITERATORS_HPP
#define LETTUCE_ITERATORS_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
// ...
//----------------------------------------
// Standard C headers
#include <cstddef>
#include <cstdlib>

// Notes/ideas on a general iterator function:
// Take algorithm and lattice as arguments, and get stencil size from algorithm, and lattice lengths/layout parameters (SIMD/MPI)
// The smallest separation we want is the stencil size, so exclude all iterations that have a smaller stencil size
// Then simply search for the ideal allowed choice

namespace Iterator
{
    // Sequential iteration through the lattice, where the function is applied to each lattice link
    template<typename funcT>
    void Sequential(funcT&& function, const int n_sweep = 1)
    {
        // #pragma omp parallel
        for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
        {
            #pragma omp parallel for
            // #pragma omp for
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
            for (int mu = 0; mu < 4; ++mu)
            {
                function({t, x, y, z, mu});
            }
        }
    }
    //-----
    // Sequential iteration through the lattice, where the function is applied to each lattice link and the return value is summed up
    template<typename funcT, typename sumT>
    void SequentialSum(funcT&& function, sumT& sum, const int n_sweep = 1)
    {
        // #pragma omp parallel
        for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
        {
            #pragma omp parallel for reduction(+: sum)
            // #pragma omp for reduction(+: sum)
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
            for (int mu = 0; mu < 4; ++mu)
            {
                sum += function({t, x, y, z, mu});
            }
        }
    }
    //-----
    // Checkerboard iteration through the lattice, where the function is applied to each lattice link
    template<typename funcT>
    void Checkerboard(funcT&& function, const int n_sweep = 1)
    {
        static_assert(Nt % 2 == 0 and Nx % 2 == 0 and Ny % 2 == 0 and Nz % 2 == 0, "Currently, only even lattice sizes are supported with parallelization.");
        // #pragma omp parallel
        for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
        for (int mu = 0; mu < 4; ++mu)
        for (int eo = 0; eo < 2; ++eo)
        {
            #pragma omp parallel for
            // #pragma omp for
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            {
                int offset {((t + x + y) & 1) ^ eo};
                for (int z = offset; z < Nz; z+=2)
                {
                    function({t, x, y, z, mu});
                }
            }
        }
    }
    //-----
    // Checkerboard iteration through the lattice, where the function is applied to each lattice link and the return value is summed up
    template<typename funcT, typename sumT>
    void CheckerboardSum(funcT&& function, sumT& sum, const int n_sweep = 1)
    {
        static_assert(Nt % 2 == 0 and Nx % 2 == 0 and Ny % 2 == 0 and Nz % 2 == 0, "Currently, only even lattice sizes are supported with parallelization.");
        // #pragma omp parallel
        for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
        for (int mu = 0; mu < 4; ++mu)
        for (int eo = 0; eo < 2; ++eo)
        {
            #pragma omp parallel for reduction(+: sum)
            // #pragma omp for reduction(+: sum)
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            {
                int offset {((t + x + y) & 1) ^ eo};
                for (int z = offset; z < Nz; z+=2)
                {
                    sum += function({t, x, y, z, mu});
                }
            }
        }
    }
    // //-----
    // // TODO: Implement!
    // // Checkerboard-like iteration through the lattice (for rectangular 2x1 actions), where the function is applied to each lattice link
    // template<funcT>
    // void Checkerboard3(funcT&& function, const int n_sweep = 1)
    // {
    //     // TODO: 3 or 6?
    //     static_assert(Nt % 3 == 0 and Nx % 3 == 0 and Ny % 3 == 0 and Nz % 3 == 0, "Currently, only lattice sizes divisible by 3 are supported with parallelization.");
    //     for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
    //     for (int mu = 0; mu < 4; ++mu)
    //     // For each direction, the lattice is split up into 6 sublattices that must be update one after another
    //     for (int sublat = 0; sublat < 6; ++sublat)
    //     {
    //         #pragma omp parallel for
    //         for (int t = 0; t < Nt; ++t)
    //         for (int x = 0; x < Nx; ++x)
    //         for (int y = 0; y < Ny; ++y)
    //         {
    //             int offset {((t + x + y) & 1) ^ eo};
    //             for (int z = offset; z < Nz; z+=2)
    //             {
    //                 function({t, x, y, z, mu});
    //             }
    //         }
    //     }
    // }
    //-----
    // Checkerboard-like iteration through the lattice (for rectangular 2x1 actions), where the function is applied to each lattice link
    template<typename funcT>
    void Checkerboard4(funcT&& function, const int n_sweep = 1)
    {
        // TODO: If checked like this, the static_assert complains, even if the function is never used. Need to explicitly check on an array that we pass to the function...
        // static_assert(Nt % 4 == 0 and Nx % 4 == 0 and Ny % 4 == 0 and Nz % 4 == 0, "Currently, only lattice sizes divisible by 4 are supported with parallelization.");
        if constexpr(Nt % 4 != 0 or Nx % 4 != 0 or Ny % 4 != 0 or Nz % 4 != 0)
        {
            std::cerr << "Attempted to use Checkerboard4 function with a lattice where at least one length is not divisible by 4!" << std::endl;
            std::exit(1);
        }
        for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
        for (int mu = 0; mu < 4; ++mu)
        // For each direction, the lattice is split up into 4 sublattices that must be update one after another
        for (int sublat = 0; sublat < 4; ++sublat)
        {
            #pragma omp parallel for
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
            {
                link_coord current_link {t, x, y, z, mu};
                if ((current_link.sum() + current_link[mu]) % 4 == sublat)
                {
                    function({t, x, y, z, mu});
                }
            }
        }
    }
    //-----
    template<typename funcT, typename sumT>
    void Checkerboard4Sum(funcT&& function, sumT& sum, const int n_sweep = 1)
    {
        // TODO: If checked like this, the static_assert complains, even if the function is never used. Need to explicitly check on an array that we pass to the function...
        // static_assert(Nt % 4 == 0 and Nx % 4 == 0 and Ny % 4 == 0 and Nz % 4 == 0, "Currently, only lattice sizes divisible by 4 are supported with parallelization.");
        if constexpr(Nt % 4 != 0 or Nx % 4 != 0 or Ny % 4 != 0 or Nz % 4 != 0)
        {
            std::cerr << "Attempted to use Checkerboard4 function with a lattice where at least one length is not divisible by 4!" << std::endl;
            std::exit(1);
        }
        for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
        for (int mu = 0; mu < 4; ++mu)
        // For each direction, the lattice is split up into 4 sublattices that must be update one after another
        for (int sublat = 0; sublat < 4; ++sublat)
        {
            #pragma omp parallel for reduction(+: sum)
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
            {
                link_coord current_link {t, x, y, z, mu};
                if ((current_link.sum() + current_link[mu]) % 4 == sublat)
                {
                    sum += function({t, x, y, z, mu});
                }
            }
        }
    }
    //-----
    // Random sequential iterator through the lattice
    template<typename funcT>
    void Random(funcT&& function, const int n_sweep = 1)
    {
        std::uniform_int_distribution<> dist_t(0, Nt - 1);
        std::uniform_int_distribution<> dist_x(0, Nx - 1);
        std::uniform_int_distribution<> dist_y(0, Ny - 1);
        std::uniform_int_distribution<> dist_z(0, Nz - 1);
        std::uniform_int_distribution<> dist_mu(0, 4 - 1);
        // #pragma omp parallel
        for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
        {
            // #pragma omp parallel for
            // #pragma omp parallel
            for (std::size_t i = 0; i < Nt * Nx * Ny * Nz; ++i)
            {
                link_coord current_link {dist_t(generator_rand), dist_x(generator_rand), dist_y(generator_rand), dist_z(generator_rand), dist_mu(generator_rand)};
                function(current_link);
            }
        }
    }
} // namespace Iterator

#endif // LETTUCE_ITERATORS_HPP
