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
#include <cstddef>

// Struct holding the lattice
// Can be indexed with individual coordinates (t, x, y, z, mu)
// Can also take a link_coord struct as coordinates

// TODO: Split up into class which only consists of the array holding a gauge field?
//       Probably better for handling the smeared fields, since that way we can use an array/vector of n_smear gauge fields

// DO NOT EVER USE THIS, ONLY FOR INTERNAL USAGE!

template<std::size_t size_, typename gaugeT>
class GaugeFieldRaw
{
    private:
        std::unique_ptr<gaugeT[]> gaugefield_raw {std::make_unique<gaugeT[]>(size)};
    public:
        static constexpr std::size_t size {size_};
        // Default constructor (apparantly not allowed to be marked as noexcept, compiler will complain)
        GaugeFieldRaw() = default;
        // Default destructor
        ~GaugeFieldRaw() = default;
        // Copy constructor
        GaugeFieldRaw(const GaugeFieldRaw& field_in) noexcept
        {
            static_assert(size == field_in.size, "Can't construct GaugeFieldRaw from another instance with different size!");
            for (std::size_t ind = 0; ind < size; ++ind)
            {
                gaugefield_raw[ind] = field_in.gaugefield_raw[ind];
            }
        }
        // Copy assignment
        // We don't need assignment chaining, so return void instead of GaugeFieldRaw&
        void operator=(const GaugeFieldRaw& field_in) noexcept
        {
            static_assert(size == field_in.size, "Can't copy GaugeFieldRaw from another instance with different size!");
            for (std::size_t ind = 0; ind < size; ++ind)
            {
                gaugefield_raw[ind] = field_in.gaugefield_raw[ind];
            }
        }
        //
        gaugeT& operator[](const std::size_t ind) noexcept
        {
            return gaugefield_raw[ind];
        }
        gaugeT operator[](const std::size_t ind) const noexcept
        {
            return gaugefield_raw[ind];
        }
};

// template<typename gaugeT, typename LayoutT>
template<std::size_t Nt_, std::size_t Nx_, std::size_t Ny_, std::size_t Nz_, typename gaugeT>
class GaugeField4D
{
    private:
        static constexpr std::size_t Nt  {Nt_};
        static constexpr std::size_t Nx  {Nx_};
        static constexpr std::size_t Ny  {Ny_};
        static constexpr std::size_t Nz  {Nz_};
        static constexpr std::size_t Nmu {4};
        static constexpr std::size_t V   {Nt * Nx * Ny * Nz * Nmu};
        GaugeFieldRaw<V, gaugeT> gaugefield;
    public:
        // Constructor with four arguments (one length for each direction)
        GaugeField4D() noexcept
            {
                std::cout << "Creating GaugeField4D with volume: " << V << std::endl;
            }
        // // Delete default constructor
        // GaugeField4D() = delete;
        // Destructor
        ~GaugeField4D()
        {
            std::cout << "Deleting GaugeField4D with volume: " << V << std::endl;
        }
        // Copy constructor
        // GaugeField4D(const GaugeField4D& field_in) noexcept :
        //     Nt(field_in.Nt), Nx(field_in.Nx), Ny(field_in.Ny), Nz(field_in.Nz), V(field_in.Nt * field_in.Nx * field_in.Ny * field_in.Nz * field_in.Nmu)
        //     {
        //         #pragma omp parallel for
        //         for (int t = 0; t < Nt; ++t)
        //         for (int x = 0; x < Nx; ++x)
        //         for (int y = 0; y < Ny; ++y)
        //         for (int z = 0; z < Nz; ++z)
        //         for (int mu = 0; mu < Nmu; ++mu)
        //         {
        //             gaugefield[LinearCoordinate(t, x, y, z, mu)] = field_in.gaugefield[LinearCoordinate(t, x, y, z, mu)];
        //         }
        //     }
        // Copy assignment
        // We don't need assignment chaining, so return void instead of GaugeField4D&
        // TODO: Is this okay? Correctness, performance?
        void operator=(const GaugeField4D& field_in) noexcept
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
        gaugeT& operator()(const std::size_t linear_coord) noexcept
        {
            return gaugefield[linear_coord];
        }
        gaugeT operator()(const std::size_t linear_coord) const noexcept
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
        std::size_t Volume() const noexcept
        {
            return V;
        }
        std::size_t Length(const int direction) const noexcept
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
        inline std::size_t LinearCoordinate(const site_coord& site, const int mu) const noexcept
        {
            return (((site.t * Nx + site.x) * Ny + site.y) * Nz + site.z) * Nmu + mu;
        }
        [[nodiscard]]
        inline std::size_t LinearCoordinate(const link_coord& coord) const noexcept
        {
            return (((coord.t * Nx + coord.x) * Ny + coord.y) * Nz + coord.z) * Nmu + coord.mu;
        }
        [[nodiscard]]
        inline std::size_t LinearCoordinate(const int t, const int x, const int y, const int z, const int mu) const noexcept
        {
            return (((t * Nx + x) * Ny + y) * Nz + z) * Nmu + mu;
        }
        // Transform 5 integers into linear coordinate (direction is the slowest index)
        // int LinearCoordinate(const int t, const int x, const int y, const int z, const int mu) const noexcept
        // {
        //     return (((mu * Nmu + t) * Nx + x) * Ny + y) * Nz + z;
        // }
};

template<std::size_t Nt_, std::size_t Nx_, std::size_t Ny_, std::size_t Nz_, typename gaugeT>
class GaugeField4DSmeared
{
    private:
        const  int                   Nsmear;
        static constexpr std::size_t Nt  {Nt_};
        static constexpr std::size_t Nx  {Nx_};
        static constexpr std::size_t Ny  {Ny_};
        static constexpr std::size_t Nz  {Nz_};
        static constexpr std::size_t Nmu {4};
        static constexpr std::size_t V   {Nt * Nx * Ny * Nz * Nmu};
        std::unique_ptr<GaugeFieldRaw<V, gaugeT>[]> gaugefield {std::make_unique<GaugeFieldRaw<V, gaugeT>[]>(Nsmear)};
    public:
        // Constructor with Nsmear as argument
        GaugeField4DSmeared(const int Nsmear_in) noexcept :
            Nsmear(Nsmear_in)
            {
                std::cout << "Creating GaugeField4DSmeared (Nsmear = " << Nsmear << ") with volume: " << V << std::endl;
                // gaugefield.resize(V);
            }
        // Delete default constructor
        GaugeField4DSmeared() = delete;
        // Destructor
        ~GaugeField4DSmeared()
        {
            std::cout << "Deleting GaugeField4DSmeared (Nsmear = " << Nsmear << ") with volume: " << V << std::endl;
        }
        // Overload operator for access to individual smearing levels
        [[nodiscard]]
        GaugeFieldRaw<V, gaugeT>& operator[](const int n) noexcept
        {
            return gaugefield[n];
        }
        [[nodiscard]]
        GaugeFieldRaw<V, gaugeT> operator[](const int n) const noexcept
        {
            return gaugefield[n];
        }
};

using GaugeField = GaugeField4D<Nt, Nx, Ny, Nz, Matrix_SU3>;

#endif // LETTUCE_LATTICE_HPP
