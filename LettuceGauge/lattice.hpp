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
#include <type_traits>
//----------------------------------------
// Standard C headers
#include <cstddef>

//----------------------------------------

// This class is a minimal wrapper around an array containing a gauge field, to be used in the GaugeField4D and GaugeField4DSmeared classes
// It should never be used by itself, hence everything is private
template<std::size_t size_, typename T>
class LatticeField
{
    private:
        // This class should only be used internally as part of a wrapper class, so everything is private
        template<int Nt_, int Nx_, int Ny_, int Nz_, typename S>
        friend class GaugeField4D;

        template<int Nt_, int Nx_, int Ny_, int Nz_, typename S>
        friend class GaugeField4DSmeared;

        template<int Nt_, int Nx_, int Ny_, int Nz_, typename S>
        friend class FullTensor4D;

        std::unique_ptr<T[]> buffer {std::make_unique<T[]>(size)};
        static constexpr std::size_t size {size_};

        LatticeField() noexcept = default;

        ~LatticeField() = default;

        LatticeField(const LatticeField& field_in) noexcept
        {
            // std::cout << "Copy constructor of LatticeField used!" << std::endl;
            #pragma omp parallel for
            for (std::size_t ind = 0; ind < size; ++ind)
            {
                buffer[ind] = field_in.buffer[ind];
            }
        }

        LatticeField& operator=(const LatticeField& field_in) noexcept
        {
            // std::cout << "Copy assignment of LatticeField used!" << std::endl;
            #pragma omp parallel for
            for (std::size_t ind = 0; ind < size; ++ind)
            {
                buffer[ind] = field_in.buffer[ind];
            }
            return *this;
        }

        T& operator[](const std::size_t ind) noexcept
        {
            return buffer[ind];
        }

        const T& operator[](const std::size_t ind) const noexcept
        {
            return buffer[ind];
        }

        // Implement swap by swapping the underlying pointers
        friend void Swap(LatticeField& field1, LatticeField& field2) noexcept
        {
            field1.buffer.swap(field2.buffer);
            // std::swap(field1.buffer, field2.buffer);
            // auto tmp = std::move(field1.buffer);
            // field1.buffer = std::move(field2.buffer);
            // field2.buffer = std::move(tmp);
        }
};

// This class acts as a general container for gauge fields in 4 dimensions
// The links can be accessed via a single lexicographic index, link_coords, or site_coords and an additional directional index
template<int Nt_, int Nx_, int Ny_, int Nz_, typename gaugeT>
class GaugeField4D
{
    private:
        static constexpr int         Nt     {Nt_};
        static constexpr int         Nx     {Nx_};
        static constexpr int         Ny     {Ny_};
        static constexpr int         Nz     {Nz_};
        static constexpr int         Nmu    {4};
        // Promote single length to size_t so the product doesn't overflow even if std::size_t would be big enough
        static constexpr std::size_t V      {static_cast<std::size_t>(Nt) * Nx * Ny * Nz};
        static constexpr std::size_t V_link {Nmu * V};
        LatticeField<V_link, gaugeT> gaugefield;
    public:
        GaugeField4D() noexcept
        {
            std::cout << "Creating GaugeField4D with volume: " << V << std::endl;
        }
        // // Delete default constructor
        // GaugeField4D() = delete;

        ~GaugeField4D()
        {
            std::cout << "Deleting GaugeField4D with volume: " << V << std::endl;
        }
        // Copy constructor
        // GaugeField4D(const GaugeField4D& field_in) noexcept = default;
        // GaugeField4D(const GaugeField4D& field_in) noexcept :
        //     Nt(field_in.Nt), Nx(field_in.Nx), Ny(field_in.Ny), Nz(field_in.Nz), V(field_in.Nt * field_in.Nx * field_in.Ny * field_in.Nz * field_in.Nmu)
        //     {
        //         #pragma omp parallel for collapse(2)
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
        GaugeField4D& operator=(const GaugeField4D& field_in) noexcept
        {
            // std::cout << "Copy assignment operator of GaugeField4D used" << std::endl;
            // Check for self-assignments
            if (this != &field_in)
            {
                gaugefield = field_in.gaugefield;
            }
            return *this;
        }
        //-----
        // Access gauge links via single integer coordinate
        gaugeT& operator()(const std::size_t linear_coord) noexcept
        {
            return gaugefield[linear_coord];
        }
        const gaugeT& operator()(const std::size_t linear_coord) const noexcept
        {
            return gaugefield[linear_coord];
        }
        // -----
        // Access gauge links via site_coord and direction
        gaugeT& operator()(const site_coord& site, const int mu) noexcept
        {
            return gaugefield[LinearCoordinate(site, mu)];
        }
        const gaugeT& operator()(const site_coord& site, const int mu) const noexcept
        {
            return gaugefield[LinearCoordinate(site, mu)];
        }
        // -----
        // Access gauge links via link_coord
        gaugeT& operator()(const link_coord& coord) noexcept
        {
            return gaugefield[LinearCoordinate(coord)];
        }
        const gaugeT& operator()(const link_coord& coord) const noexcept
        {
            return gaugefield[LinearCoordinate(coord)];
        }

        void SetToIdentity() noexcept
        {
            #pragma omp parallel for collapse(2)
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
            for (int mu = 0; mu < Nmu; ++mu)
            {
                gaugefield[LinearCoordinate(t, x, y, z, mu)].setIdentity();
            }
            std::cout << "Gauge Fields set to identity!" << std::endl;
        }
        void SetToZero() noexcept
        {
            #pragma omp parallel for collapse(2)
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
            for (int mu = 0; mu < Nmu; ++mu)
            {
                gaugefield[LinearCoordinate(t, x, y, z, mu)].setZero();
            }
            // std::cout << "Gauge Fields set to zero!" << std::endl;
        }

        constexpr std::size_t Volume() const noexcept
        {
            return V;
        }
        constexpr std::size_t SpatialVolume() const noexcept
        {
            return Volume() / Nt;
        }
        constexpr int Length(const int direction) const noexcept
        {
            switch (direction)
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
        constexpr site_coord Shape() const noexcept
        {
            return {Nt, Nx, Ny, Nz};
        }
        friend std::ostream& operator<<(std::ostream& stream, const GaugeField4D& field)
        {
            for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
            for (int mu = 0; mu < Nmu; ++mu)
            {
                stream << field.gaugefield[field.LinearCoordinate(t, x, y, z, mu)] << ", ";
            }
            return stream;
        }
        // We only need to swap the raw gaugefields
        friend void Swap(GaugeField4D& U1, GaugeField4D& U2) noexcept
        {
            // Shapes are always the same as long as the extents are template parameters
            // if (U1.Shape() != U2.Shape())
            // {
            //     std::cerr << "Warning: Trying to swap two arrays with different sizes!" << std::endl;
            // }
            Swap(U1.gaugefield, U2.gaugefield);
        }

        // Move site coordinates
        template<int dist>
        [[nodiscard]]
        site_coord Move(const site_coord& site, const int direction) const noexcept
        {
            // site_coord new_site = site;
            // new_site[direction] = (new_site[direction] + dist) % N[direction];
            // TODO: Check bounds? If dist < Nx, Ny, Nz, Nt this is unproblematic, but what if dist is larger than linear extent? Check at compile time?
            static_assert(dist != 0, "Move with dist == 0 detected!");
            // For positive dist we do not need to consider negative modulo
            if constexpr(dist > 0)
            {
                switch (direction)
                {
                    case 0:
                        return site_coord((site.t + dist) % Nt, site.x              , site.y              , site.z              );
                    case 1:
                        return site_coord(site.t              , (site.x + dist) % Nx, site.y              , site.z              );
                    case 2:
                        return site_coord(site.t              , site.x              , (site.y + dist) % Ny, site.z              );
                    case 3:
                        return site_coord(site.t              , site.x              , site.y              , (site.z + dist) % Nz);
                    default:
                        return site_coord(site.t              , site.x              , site.y              , site.z              );
                }
            }
            else
            {
                // TODO: Perhaps replace with safer (potentially less efficient) version?
                static_assert(dist <= Nt and dist <= Nx and dist <= Ny and dist <= Nz, "Move in negative direction with dist greater than one of the lattice lengths detected!");
                // Alternative: Use negative indices (how to deal with 0 then?)
                switch (direction)
                {
                    case 0:
                        return site_coord((site.t + Nt + dist) % Nt, site.x                   , site.y                   , site.z                   );
                    case 1:
                        return site_coord(site.t                   , (site.x + Nx + dist) % Nx, site.y                   , site.z                   );
                    case 2:
                        return site_coord(site.t                   , site.x                   , (site.y + Ny + dist) % Ny, site.z                   );
                    case 3:
                        return site_coord(site.t                   , site.x                   , site.y                   , (site.z + Nz + dist) % Nz);
                    default:
                        return site_coord(site.t                   , site.x                   , site.y                   , site.z                   );
                }
            }
        }

        // Move link coordinates
        template<int dist>
        [[nodiscard]]
        link_coord Move(const link_coord& link, const int direction) const noexcept
        {
            // link_coord new_link = link;
            // new_link[direction] = (new_link[direction] + dist) % N[direction];
            // TODO: Check bounds? If dist < Nx, Ny, Nz, Nt this is unproblematic, but what if dist is larger than linear extent? Check at compile time?
            static_assert(dist != 0, "Move with dist == 0 detected!");
            // For positive dist we do not need to consider negative modulo
            if constexpr(dist > 0)
            {
                switch (direction)
                {
                    case 0:
                        return link_coord((link.t + dist) % Nt, link.x              , link.y              , link.z              , link.mu);
                    case 1:
                        return link_coord(link.t              , (link.x + dist) % Nx, link.y              , link.z              , link.mu);
                    case 2:
                        return link_coord(link.t              , link.x              , (link.y + dist) % Ny, link.z              , link.mu);
                    case 3:
                        return link_coord(link.t              , link.x              , link.y              , (link.z + dist) % Nz, link.mu);
                    default:
                        return link_coord(link.t              , link.x              , link.y              , link.z              , link.mu);
                }
            }
            else
            {
                // TODO: Perhaps replace with safer (potentially less efficient) version?
                static_assert(dist <= Nt and dist <= Nx and dist <= Ny and dist <= Nz, "Move in negative direction with dist greater than one of the lattice lengths detected!");
                // Alternative: Use negative indices (how to deal with 0 then?)
                switch (direction)
                {
                    case 0:
                        return link_coord((link.t + Nt + dist) % Nt, link.x                   , link.y                   , link.z                   , link.mu);
                    case 1:
                        return link_coord(link.t                   , (link.x + Nx + dist) % Nx, link.y                   , link.z                   , link.mu);
                    case 2:
                        return link_coord(link.t                   , link.x                   , (link.y + Ny + dist) % Ny, link.z                   , link.mu);
                    case 3:
                        return link_coord(link.t                   , link.x                   , link.y                   , (link.z + Nz + dist) % Nz, link.mu);
                    default:
                        return link_coord(link.t                   , link.x                   , link.y                   , link.z                   , link.mu);
                }
            }
        }
        // Non-templated version?
        // link_coord Move(link_coord& link, const int dir, const int dist = 1) const noexcept
        // {
        //     // link_coord new_link = link;
        //     // new_link[dir] = (new_link[dir] + dist) % N[dir];
        //     switch (dir)
        //     {
        //         case 0:
        //             return link_coord((link.t + Nt + dist)%Nt, link.x                 , link.y                 , link.z                 , link.mu);
        //         case 1:
        //             return link_coord(link.t                 , (link.x + Nx + dist)%Nx, link.y                 , link.z                 , link.mu);
        //         case 2:
        //             return link_coord(link.t                 , link.x                 , (link.y + Ny + dist)%Ny, link.z                 , link.mu);
        //         case 3:
        //             return link_coord(link.t                 , link.x                 , link.y                 , (link.z + Nz + dist)%Nz, link.mu);
        //         default:
        //             return link_coord(link.t                 , link.x                 , link.y                 , link.z                 , link.mu)
        //     }
        // }
    private:
        // -----
        // TODO: Do we need modulo here? Also, it is probably preferable to make the layout/coordinate function a (template) parameter of the class
        //       For lattice lengths that are powers of two, we can replace x%Nx by x&(Nx-1) (possibly faster?)
        // Transform 5 integers into linear coordinate (direction is the fastest index)
        [[nodiscard]]
        inline std::size_t LinearCoordinate(const site_coord& site, const int mu) const noexcept
        {
            // Promote single length to size_t so the product doesn't overflow
            return (((site.t * static_cast<std::size_t>(Nx) + site.x) * Ny + site.y) * Nz + site.z) * Nmu + mu;
        }
        [[nodiscard]]
        inline std::size_t LinearCoordinate(const link_coord& coord) const noexcept
        {
            // Promote single length to size_t so the product doesn't overflow
            return (((coord.t * static_cast<std::size_t>(Nx) + coord.x) * Ny + coord.y) * Nz + coord.z) * Nmu + coord.mu.direction;
        }
        [[nodiscard]]
        inline std::size_t LinearCoordinate(const int t, const int x, const int y, const int z, const int mu) const noexcept
        {
            // Promote single length to size_t so the product doesn't overflow
            return (((t * static_cast<std::size_t>(Nx) + x) * Ny + y) * Nz + z) * Nmu + mu;
        }
        // Transform 5 integers into linear coordinate (direction is the slowest index)
        // int LinearCoordinate(const int t, const int x, const int y, const int z, const int mu) const noexcept
        // {
        //     // Promote single length to size_t so the product doesn't overflow
        //     return (((mu * static_cast<std::size_t>(Nmu) + t) * Nx + x) * Ny + y) * Nz + z;
        // }
        // Inverse of functions above, transform linear coordinate to tuple of coordinates/5 integers
        // link_coord LinkCoordinate(std::size_t ind) const noexcept
        // {
        //     return {(ind / (Nmu * Nz * Ny * Nx)) % Nt, (ind / (Nmu * Nz * Ny)) % Nx, (ind / (Nmu * Nz)) % Ny, (ind / Nmu) % Nz, static_cast<int>(ind % Nmu)};
        // }
};

// This class acts as a general container for multiple gauge fields in 4 dimensions (mainly meant to be used for smearing/calculation of smeared forces)
// The [] operator provides access to the different smearing levels, i.e., it returns a reference to a GaugeField4D (which can the be accessed and manipulated in the usual way)
template<int Nt_, int Nx_, int Ny_, int Nz_, typename gaugeT>
class GaugeField4DSmeared
{
    private:
        const  int                   Nsmear;
        static constexpr int         Nt     {Nt_};
        static constexpr int         Nx     {Nx_};
        static constexpr int         Ny     {Ny_};
        static constexpr int         Nz     {Nz_};
        static constexpr int         Nmu    {4};
        // Promote single length to size_t so the product doesn't overflow
        static constexpr std::size_t V      {static_cast<std::size_t>(Nt) * Nx * Ny * Nz};
        static constexpr std::size_t V_link {Nmu * V};
        std::unique_ptr<GaugeField4D<Nt, Nx, Ny, Nz, gaugeT>[]> gaugefield {std::make_unique<GaugeField4D<Nt, Nx, Ny, Nz, gaugeT>[]>(Nsmear)};
    public:
        // Constructor with Nsmear as argument
        explicit GaugeField4DSmeared(const int Nsmear_in) noexcept :
        Nsmear(Nsmear_in)
        {
            std::cout << "Creating GaugeField4DSmeared (Nsmear = " << Nsmear << ") with volume: " << V << std::endl;
            // gaugefield.resize(V);
        }

        GaugeField4DSmeared() = delete;

        ~GaugeField4DSmeared()
        {
            std::cout << "Deleting GaugeField4DSmeared (Nsmear = " << Nsmear << ") with volume: " << V << std::endl;
        }

        // Overload operator for access to individual smearing levels
        [[nodiscard]]
        GaugeField4D<Nt, Nx, Ny, Nz, gaugeT>& operator[](const int n) noexcept
        {
            return gaugefield[n];
        }
        [[nodiscard]]
        const GaugeField4D<Nt, Nx, Ny, Nz, gaugeT>& operator[](const int n) const noexcept
        {
            return gaugefield[n];
        }
        int ReturnNsmear() const noexcept
        {
            return Nsmear;
        }
};

// This class acts as a general container for a (4x4)-component tensor in 4 dimensions
// The links can be accessed via a single lexicographic index, link_coords, or site_coords and an additional directional index
template<int Nt_, int Nx_, int Ny_, int Nz_, typename gaugeT>
class FullTensor4D
{
    private:
        static constexpr int         Nt     {Nt_};
        static constexpr int         Nx     {Nx_};
        static constexpr int         Ny     {Ny_};
        static constexpr int         Nz     {Nz_};
        static constexpr int         Nmu    {4};
        static constexpr int         Nnu    {4};
        // Promote single length to size_t so the product doesn't overflow
        static constexpr std::size_t V      {static_cast<std::size_t>(Nt) * Nx * Ny * Nz};
        static constexpr std::size_t V_link {Nmu * Nnu * V};
        LatticeField<V_link, gaugeT> gaugefield;
    public:
        // Constructor with four arguments (one length for each direction)
        FullTensor4D() noexcept
        {
            std::cout << "Creating FullTensor4D with volume: " << V << std::endl;
        }

        ~FullTensor4D()
        {
            std::cout << "Deleting FullTensor4D with volume: " << V << std::endl;
        }

        // Copy assignment
        FullTensor4D& operator=(const FullTensor4D& field_in) noexcept
        {
            // std::cout << "Copy assignment operator of FullTensor4D used" << std::endl;
            // Check for self-assignments
            if (this != &field_in)
            {
                // Copy using OpenMP seems to be faster than single-threaded std::copy (at least for "larger" 32^4 lattices)
                #pragma omp parallel for collapse(2)
                for (int t = 0; t < Nt; ++t)
                for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z)
                for (int mu = 0; mu < Nmu; ++mu)
                for (int nu = 0; nu < Nnu; ++nu)
                {
                    gaugefield[LinearCoordinate(t, x, y, z, mu, nu)] = field_in.gaugefield[LinearCoordinate(t, x, y, z, mu, nu)];
                }
                // gaugefield = field_in.gaugefield;
            }
            return *this;
        }
        //-----
        // Access gauge links via single integer coordinate
        gaugeT& operator()(const std::size_t linear_coord) noexcept
        {
            return gaugefield[linear_coord];
        }
        const gaugeT& operator()(const std::size_t linear_coord) const noexcept
        {
            return gaugefield[linear_coord];
        }
        // -----
        // Access gauge links via site_coord and two directions
        gaugeT& operator()(const site_coord& site, const int mu, const int nu) noexcept
        {
            return gaugefield[LinearCoordinate(site, mu, nu)];
        }
        const gaugeT& operator()(const site_coord& site, const int mu, const int nu) const noexcept
        {
            return gaugefield[LinearCoordinate(site, mu, nu)];
        }
        // -----
        // Access gauge links via link_coord
        // gaugeT& operator()(const link_coord& coord) noexcept
        // {
        //     return gaugefield[LinearCoordinate(coord)];
        // }
        // gaugeT operator()(const link_coord& coord) const noexcept
        // {
        //     return gaugefield[LinearCoordinate(coord)];
        // }
        // Access gauge links via 6 ints
        // [[deprecated("Using individual coordinates is disencouraged, use link_coord instead")]]
        gaugeT& operator()(const int t, const int x, const int y, const int z, const int mu, const int nu) noexcept
        {
            return gaugefield[LinearCoordinate(t, x, y, z, mu, nu)];
        }
        // [[deprecated("Using individual coordinates is disencouraged, use link_coord instead")]]
        const gaugeT& operator()(const int t, const int x, const int y, const int z, const int mu, const int nu) const noexcept
        {
            return gaugefield[LinearCoordinate(t, x, y, z, mu, nu)];
        }
        constexpr std::size_t Volume() const noexcept
        {
            return V;
        }
        constexpr std::size_t SpatialVolume() const noexcept
        {
            return Volume() / Nt;
        }
        constexpr int Length(const int direction) const noexcept
        {
            switch (direction)
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
        // TODO: Should we add a Shape() function here like for GaugeField4D?
    private:
        // -----
        // TODO: Do we need modulo here? Also, it is probably preferable to make the layout/coordinate function a (template) parameter of the class
        //       For lattice lengths that are powers of two, we can replace x%Nx by x&(Nx-1) (possibly faster?)
        // Transform 5 integers into linear coordinate (direction is the fastest index)
        [[nodiscard]]
        inline std::size_t LinearCoordinate(const site_coord& site, const int mu, const int nu) const noexcept
        {
            // Promote single length to size_t so the product doesn't overflow
            return ((((site.t * static_cast<std::size_t>(Nx) + site.x) * Ny + site.y) * Nz + site.z) * Nmu + mu) * Nnu + nu;
        }
        // [[nodiscard]]
        // inline std::size_t LinearCoordinate(const link_coord& coord) const noexcept
        // {
        //     // Promote single length to size_t so the product doesn't overflow
        //     return (((coord.t * static_cast<std::size_t>(Nx) + coord.x) * Ny + coord.y) * Nz + coord.z) * Nmu + coord.mu.direction;
        // }
        [[nodiscard]]
        inline std::size_t LinearCoordinate(const int t, const int x, const int y, const int z, const int mu, const int nu) const noexcept
        {
            // Promote single length to size_t so the product doesn't overflow
            return ((((t * static_cast<std::size_t>(Nx) + x) * Ny + y) * Nz + z) * Nmu + mu) * Nnu + nu;
        }
        // Transform 5 integers into linear coordinate (direction is the slowest index)
        // int LinearCoordinate(const int t, const int x, const int y, const int z, const int mu) const noexcept
        // {
        //     // Promote single length to size_t so the product doesn't overflow
        //     return (((mu * static_cast<std::size_t>(Nmu) + t) * Nx + x) * Ny + y) * Nz + z;
        // }
};

using GaugeField        = GaugeField4D<Nt, Nx, Ny, Nz, Matrix_SU3>;
using GaugeFieldSmeared = GaugeField4DSmeared<Nt, Nx, Ny, Nz, Matrix_SU3>;
using FullTensor        = FullTensor4D<Nt, Nx, Ny, Nz, Matrix_SU3>;

// Struct to hold a pair of references to smeared fields
// Useful when smearing multiple times, and only the final smearing level is needed
// TODO: Make into template and move above type aliases?
// struct SmearedFieldTuple
// {
//     GaugeField& Field1;
//     GaugeField& Field2;
//     SmearedFieldTuple(GaugeField& Field1_in, GaugeField& Field2_in) noexcept :
//     Field1(Field1_in), Field2(Field2_in)
//     {}
//     ~SmearedFieldTuple() = default;
//     SmearedFieldTuple(const SmearedFieldTuple& tuple_in) noexcept :
//     Field1(tuple_in.Field1), Field2(tuple_in.Field2)
//     {}
//     void operator=(const SmearedFieldTuple& tuple_in) noexcept
//     {
//         if (this != &tuple_in)
//         {
//             Field1 = tuple_in.Field1;
//             Field2 = tuple_in.Field2;
//         }
//     }
// };

#endif // LETTUCE_LATTICE_HPP
