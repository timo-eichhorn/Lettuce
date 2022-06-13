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

struct site_coord
{
    // TODO: Replace with array instead of ints?
    int t;
    int x;
    int y;
    int z;
    site_coord(const int t_in, const int x_in, const int y_in, const int z_in) noexcept :
        t(t_in), x(x_in), y(y_in), z(z_in)
        {}
    inline int& operator[](const int i) noexcept
    {
        switch(i)
        {
            case 0:
                return t;
            case 1:
                return x;
            case 2:
                return y;
            case 3:
                return z;
            // TODO: Compiler will probably complain that we have no default case?
        }
    }
    inline int operator[](const int i) const noexcept
    {
        switch(i)
        {
            case 0:
                return t;
            case 1:
                return x;
            case 2:
                return y;
            case 3:
                return z;
            // TODO: Compiler will probably complain that we have no default case?
        }
    }
    friend std::ostream& operator<<(std::ostream& stream, const site_coord& site)
    {
        stream << "Site(" << site.t << ", " << site.x << ", " << site.y << ", " << site.z << ")\n";
        return stream;
    }
};

struct link_coord
{
    // TODO: Replace with array instead of ints?
    int t;
    int x;
    int y;
    int z;
    int mu;
    link_coord(const int t_in, const int x_in, const int y_in, const int z_in, const int mu_in) noexcept :
        t(t_in), x(x_in), y(y_in), z(z_in), mu(mu_in)
        {}
    inline int& operator[](const int i) noexcept
    {
        switch(i)
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
                return mu;
            // TODO: Compiler will probably complain that we have no default case?
        }
    }
    inline int operator[](const int i) const noexcept
    {
        switch(i)
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
                return mu;
            // TODO: Compiler will probably complain that we have no default case?
        }
    }
    friend std::ostream& operator<<(std::ostream& stream, const link_coord& link)
    {
        stream << "Link(" << link.t << ", " << link.x << ", " << link.y << ", " << link.z << ", " << link.mu << ")\n";
        return stream;
    }
};

// Move site coordinates
template<int dist>
[[nodiscard]]
site_coord Move(const site_coord& site, const int direction) noexcept
{
    // site_coord new_site = site;
    // new_site[direction] = (new_site[direction] + dist) % N[direction];
    // TODO: Check bounds? If dist < Nx, Ny, Nz, Nt this is unproblematic, but what if dist is larger than linear extent? Check at compile time?
    static_assert(dist != 0, "Move with dist == 0 detected!");
    // For positive dist we do not need to consider negative modulo
    if constexpr(dist > 0)
    {
        // TODO: It is probably best if we don't have to include defines.hpp to get access to Nt, Nx, ...
        //       Pass the lattice as an additional argument?
        switch(direction)
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
        // Alternative: Use negative indices (how to deal with 0 then?)
        switch(direction)
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
link_coord Move(const link_coord& link, const int direction) noexcept
{
    // link_coord new_link = link;
    // new_link[direction] = (new_link[direction] + dist) % N[direction];
    // TODO: Check bounds? If dist < Nx, Ny, Nz, Nt this is unproblematic, but what if dist is larger than linear extent? Check at compile time?
    static_assert(dist != 0, "Move with dist == 0 detected!");
    // For positive dist we do not need to consider negative modulo
    if constexpr(dist > 0)
    {
        // TODO: It is probably best if we don't have to include defines.hpp to get access to Nt, Nx, ...
        //       Pass the lattice as an additional argument?
        switch(direction)
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
        // Alternative: Use negative indices (how to deal with 0 then?)
        switch(direction)
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
// link_coord Move(link_coord& link, const int dir, const int dist = 1) noexcept
// {
//     // link_coord new_link = link;
//     // new_link[dir] = (new_link[dir] + dist) % N[dir];
//     switch(dir)
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

#endif // LETTUCE_COORDS_HPP
