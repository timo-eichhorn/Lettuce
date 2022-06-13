#ifndef LETTUCE_LATTICE_HPP
#define LETTUCE_LATTICE_HPP

// Non-standard library headers
#include "coords.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <iostream>
#include <memory>
//----------------------------------------
// Standard C headers
// ...

// Struct holding the lattice
// Can be indexed with individual coordinates (t, x, y, z, mu)
// Can also take a link_coord struct as coordinates

// template<typename gaugeT, typename LayoutT>
template<typename gaugeT>
class GaugeField4D
{
    private:
        const int           Nt;
        const int           Nx;
        const int           Ny;
        const int           Nz;
        const int           Nmu {4};
        const uint_fast32_t V;
        std::unique_ptr<gaugeT[]> gaugefield {std::make_unique<gaugeT[]>(Nt * Nx * Ny * Nz * Nmu)};
        // std::vector<gaugeT> gaugefield;
    public:
        // Constructor with four arguments (one length for each direction)
        GaugeField4D(const int Nt_in, const int Nx_in, const int Ny_in, const int Nz_in) noexcept :
            Nt(Nt_in), Nx(Nx_in), Ny(Ny_in), Nz(Nz_in), V(Nt_in * Nx_in * Ny_in * Nz_in * Nmu)
            {
                std::cout << "Creating GaugeField4D with volume: " << V << std::endl;
                // gaugefield.resize(V);
            }
        // Delete default constructor
        GaugeField4D() = delete;
        // Destructor
        ~GaugeField4D()
        {
            std::cout << "Deleting GaugeField4D with volume: " << V << std::endl;
        }
        // Copy constructor
        GaugeField4D(const GaugeField4D& field_in) noexcept :
            Nt(field_in.Nt), Nx(field_in.Nx), Ny(field_in.Ny), Nz(field_in.Nz), V(field_in.Nt * field_in.Nx * field_in.Ny * field_in.Nz * field_in.Nmu)
            {
                #pragma omp parallel for
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int mu = 0; mu < Nmu; ++mu)
                {
                    gaugefield[LinearCoordinate(t, x, y, z, mu)] = field_in.gaugefield[LinearCoordinate(t, x, y, z, mu)];
                }
            }
        // Copy assignment
        // TODO: Is this okay? Correctness, performance?
        void operator=(const GaugeField4D& field_in)
        {
            // Check for self-assignments
            if (this != &field_in)
            {
                // Check for compatible sizes
                if (Nt != field_in.Nt or Nx != field_in.Nx or Ny != field_in.Ny or Nz != field_in.Nz)
                {
                    std::cerr << "Warning: Trying to use copy assignment operator on two arrays with different sizes!" << std::endl;
                }
                // Copy
                #pragma omp parallel for
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int mu = 0; mu < Nmu; ++mu)
                {
                    gaugefield[LinearCoordinate(t, x, y, z, mu)] = field_in.gaugefield[LinearCoordinate(t, x, y, z, mu)];
                }
            }
        }
        //-----
        // Access gauge links via single integer coordinate
        gaugeT& operator()(const uint_fast32_t linear_coord) noexcept
        {
            return gaugefield[linear_coord];
        }
        gaugeT operator()(const uint_fast32_t linear_coord) const noexcept
        {
            return gaugefield[linear_coord];
        }
        // -----
        // Access gauge links via site_coord and direction
        gaugeT& operator()(const site_coord& site, const int mu) noexcept
        {
            return gaugefield[LinearCoordinate(site, mu)];
        }
        gaugeT operator()(const site_coord& site, const int mu) const noexcept
        {
            return gaugefield[LinearCoordinate(site, mu)];
        }
        // -----
        // Access gauge links via link_coord
        gaugeT& operator()(const link_coord& coord) noexcept
        {
            return gaugefield[LinearCoordinate(coord)];
        }
        gaugeT operator()(const link_coord& coord) const noexcept
        {
            return gaugefield[LinearCoordinate(coord)];
        }
        // Access gauge links via 5 ints
        [[deprecated("Using individual coordinates is disencouraged, use link_coord instead")]]
        gaugeT& operator()(const int t, const int x, const int y, const int z, const int mu) noexcept
        {
            return gaugefield[LinearCoordinate(t, x, y, z, mu)];
        }
        [[deprecated("Using individual coordinates is disencouraged, use link_coord instead")]]
        gaugeT operator()(const int t, const int x, const int y, const int z, const int mu) const noexcept
        {
            return gaugefield[LinearCoordinate(t, x, y, z, mu)];
        }
        uint_fast32_t Volume() const noexcept
        {
            return V;
        }
        int Length(const int direction) const noexcept
        {
            switch(direction)
            {
                case 0:
                    return Nt;
                case 1:
                    return Nx;
                case 2:
                    return Ny;
                case 3:
                    return Nz;
                default:
                    return 0;
            }
        }
    private:
        // -----
        // TODO: Do we need modulo here? Also, it is probably preferable to make the layout/coordinate function a (template) parameter of the class
        //       For lattice lengths that are powers of two, we can replace x%Nx by x&(Nx-1) (possibly faster?)
        // Transform 5 integers into linear coordinate (direction is the fastest index)
        [[nodiscard]]
        inline uint_fast32_t LinearCoordinate(const site_coord& site, const int mu) const noexcept
        {
            return (((site.t * Nx + site.x) * Ny + site.y) * Nz + site.z) * Nmu + mu;
        }
        [[nodiscard]]
        inline uint_fast32_t LinearCoordinate(const link_coord& coord) const noexcept
        {
            return (((coord.t * Nx + coord.x) * Ny + coord.y) * Nz + coord.z) * Nmu + coord.mu;
        }
        [[nodiscard]]
        inline uint_fast32_t LinearCoordinate(const int t, const int x, const int y, const int z, const int mu) const noexcept
        {
            return (((t * Nx + x) * Ny + y) * Nz + z) * Nmu + mu;
        }
        // Transform 5 integers into linear coordinate (direction is the slowest index)
        // int LinearCoordinate(const int t, const int x, const int y, const int z, const int mu) const noexcept
        // {
        //     return (((mu * Nmu + t) * Nx + x) * Ny + y) * Nz + z;
        // }
};

#endif // LETTUCE_LATTICE_HPP
