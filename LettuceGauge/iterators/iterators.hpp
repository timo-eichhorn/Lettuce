#ifndef LETTUCE_ITERATORS_HPP
#define LETTUCE_ITERATORS_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
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
        #pragma omp parallel
        for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
        {
            // #pragma omp parallel for
            #pragma omp for
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
        #pragma omp parallel
        for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
        {
            // #pragma omp parallel for reduction(+:sum)
            #pragma omp for reduction(+:sum)
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
        #pragma omp parallel
        for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
        for (int mu = 0; mu < 4; ++mu)
        for (int eo = 0; eo < 2; ++eo)
        {
            // #pragma omp parallel for
            #pragma omp for
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
        #pragma omp parallel
        for (int sweep_count = 0; sweep_count < n_sweep; ++sweep_count)
        for (int mu = 0; mu < 4; ++mu)
        for (int eo = 0; eo < 2; ++eo)
        {
            // #pragma omp parallel for reduction(+:sum)
            #pragma omp for reduction(+:sum)
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
}

#endif // LETTUCE_ITERATORS_HPP
