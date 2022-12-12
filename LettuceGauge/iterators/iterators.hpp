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
// ...

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
            #pragma omp parallel for reduction(+:sum)
            // #pragma omp for reduction(+:sum)
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
            #pragma omp parallel for reduction(+:sum)
            // #pragma omp for reduction(+:sum)
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
    // TODO: Check for correctness. Also, is there a better way than checking inside the loop (similar to conventional Checkerboard function above)?
    // Checkerboard-like iteration through the lattice (for rectangular 2x1 actions), where the function is applied to each lattice link
    template<typename funcT>
    void Checkerboard4(funcT&& function, const int n_sweep = 1)
    {
        static_assert(Nt % 4 == 0 and Nx % 4 == 0 and Ny % 4 == 0 and Nz % 4 == 0, "Currently, only lattice sizes divisible by 4 are supported with parallelization.");
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
        static_assert(Nt % 4 == 0 and Nx % 4 == 0 and Ny % 4 == 0 and Nz % 4 == 0, "Currently, only lattice sizes divisible by 4 are supported with parallelization.");
        for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
        for (int mu = 0; mu < 4; ++mu)
        // For each direction, the lattice is split up into 4 sublattices that must be update one after another
        for (int sublat = 0; sublat < 4; ++sublat)
        {
            #pragma omp parallel for reduction(+:sum)
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
} // namespace Iterator

#endif // LETTUCE_ITERATORS_HPP
