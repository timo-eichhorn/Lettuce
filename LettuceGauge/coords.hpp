#ifndef LETTUCE_COORDS_HPP
#define LETTUCE_COORDS_HPP

// Non-standard library headers
#include "defines.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <iostream>
#include <ostream>
//----------------------------------------
// Standard C headers
// ...

enum class link_orientation : bool {backwards = false, forwards = true};

struct lorentz_index
{
    // TODO: Use char, short, or int?
    int              direction;
    link_orientation orientation;
    // TODO: Perhaps it might be better to make NOT mark this constructor as explicit, so we can implicitly convert integral types to direction structs?
    explicit constexpr lorentz_index(const int direction_in, link_orientation orientation_in) noexcept :
    direction(direction_in), orientation(orientation_in)
    {}
    explicit constexpr lorentz_index(const int direction_in, bool orientation_forwards = true) noexcept :
    direction(direction_in), orientation(orientation_forwards ? link_orientation::forwards : link_orientation::backwards)
    {}
    [[nodiscard]]
    friend bool operator==(const lorentz_index& dir1, const lorentz_index& dir2) noexcept
    {
        return (dir1.direction == dir2.direction and dir1.orientation == dir2.orientation);
    }
};

struct site_coord
{
    // TODO: Replace with array instead of ints?
    int t;
    int x;
    int y;
    int z;
    constexpr site_coord(const int t_in, const int x_in, const int y_in, const int z_in) noexcept :
        t(t_in), x(x_in), y(y_in), z(z_in)
        {}
    inline int& operator[](const int i) noexcept
    {
        switch (i)
        {
            case 0:
                return t;
            case 1:
                return x;
            case 2:
                return y;
            case 3:
                return z;
            default:
                // TODO: C++23 introduces unreachable() in stddef.h or std::unreachable in utility, also not sure about CUDA compatibility
                // std::unreachable();
                // __builtin_unreachable();
                std::exit(EXIT_FAILURE);
        }
    }
    inline int operator[](const int i) const noexcept
    {
        switch (i)
        {
            case 0:
                return t;
            case 1:
                return x;
            case 2:
                return y;
            case 3:
                return z;
            default:
                // TODO: C++23 introduces unreachable() in stddef.h or std::unreachable in utility, also not sure about CUDA compatibility
                // std::unreachable();
                // __builtin_unreachable();
                std::exit(EXIT_FAILURE);
        }
    }
    [[nodiscard]]
    inline int sum() const noexcept
    {
        return t + x + y + z;
    }
    friend std::ostream& operator<<(std::ostream& stream, const site_coord& site)
    {
        // stream << "Site(" << site.t << ", " << site.x << ", " << site.y << ", " << site.z << ")\n";
        stream << "(" << site.t << ", " << site.x << ", " << site.y << ", " << site.z << ")";
        return stream;
    }
    [[nodiscard]]
    friend bool operator==(const site_coord& site1, const site_coord& site2) noexcept
    {
        return (site1.t == site2.t and site1.x == site2.x and site1.y == site2.y and site1.z == site2.z);
    }
    [[nodiscard]]
    friend bool operator!=(const site_coord& site1, const site_coord& site2) noexcept
    {
        return !(site1 == site2);
    }
};

struct link_coord
{
    // TODO: Replace with array instead of ints?
    int           t;
    int           x;
    int           y;
    int           z;
    lorentz_index mu;
    constexpr link_coord(const int t_in, const int x_in, const int y_in, const int z_in, const int mu_in) noexcept :
        t(t_in), x(x_in), y(y_in), z(z_in), mu(mu_in)
        {}
    constexpr link_coord(const int t_in, const int x_in, const int y_in, const int z_in, const lorentz_index mu_in) noexcept :
        t(t_in), x(x_in), y(y_in), z(z_in), mu(mu_in)
        {}
    inline int& operator[](const int i) noexcept
    {
        switch (i)
        {
            case 0:
                return t;
            case 1:
                return x;
            case 2:
                return y;
            case 3:
                return z;
            case 4:
                return mu.direction;
            default:
                // TODO: C++23 introduces unreachable() in stddef.h or std::unreachable in utility, also not sure about CUDA compatibility
                // std::unreachable();
                // __builtin_unreachable();
                std::exit(EXIT_FAILURE);
        }
    }
    inline int operator[](const int i) const noexcept
    {
        switch (i)
        {
            case 0:
                return t;
            case 1:
                return x;
            case 2:
                return y;
            case 3:
                return z;
            case 4:
                return mu.direction;
            default:
                // TODO: C++23 introduces unreachable() in stddef.h or std::unreachable in utility, also not sure about CUDA compatibility
                // std::unreachable();
                // __builtin_unreachable();
                std::exit(EXIT_FAILURE);
        }
    }
    [[nodiscard]]
    inline int sum() const noexcept
    {
        return t + x + y + z;
    }
    friend std::ostream& operator<<(std::ostream& stream, const link_coord& link)
    {
        // stream << "Link(" << link.t << ", " << link.x << ", " << link.y << ", " << link.z << ", " << link.mu << ")\n";
        stream << "(" << link.t << ", " << link.x << ", " << link.y << ", " << link.z << ", " << (static_cast<bool>(link.mu.orientation) ? "+" : "-") << link.mu.direction << ")";
        return stream;
    }
    [[nodiscard]]
    friend bool operator==(const link_coord& link1, const link_coord& link2) noexcept
    {
        return (link1.t == link2.t and link1.x == link2.x and link1.y == link2.y and link1.z == link2.z and link1.mu == link2.mu);
    }
    [[nodiscard]]
    friend bool operator!=(const link_coord& link1, const link_coord& link2) noexcept
    {
        return !(link1 == link2);
    }
};

#endif // LETTUCE_COORDS_HPP
