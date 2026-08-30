#ifndef LETTUCE_TENSOR_SYMMETRY_HPP
#define LETTUCE_TENSOR_SYMMETRY_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <complex>
#include <concepts>
#include <type_traits>
//----------------------------------------
// Standard C headers
#include <cstddef>

//----------------------------------------

namespace TensorSymmetry
{
    namespace Detail
    {
        enum class ComponentAccess : unsigned char
        {
            Direct,
            Indirect,
            ImplicitZero
        };

        struct ComponentMap
        {
            std::size_t     component;
            ComponentAccess access;
        };

        template<typename T>
        [[nodiscard]]
        T Zero() noexcept
        {
            if constexpr(requires{T::Zero();})
            {
                return T::Zero();
            }
            else
            {
                return T{};
            }
        }

        template<typename T>
        [[nodiscard]]
        T Adjoint(const T& value) noexcept
        {
            if constexpr(requires{value.adjoint();})
            {
                return T(value.adjoint());
            }
            // Real (non-complex) type
            // TODO: Better, more explicit complex check
            else if constexpr(std::is_arithmetic_v<T>)
            {
                return value;
            }
            else
            {
                return T(std::conj(value));
            }
        }

        // Shared representation for all policies where the independent components are contained within a triangle
        template<bool StoresDiagonal>
        struct UpperTriangularSymmetryRelation
        {
            template<int N>
            [[nodiscard]]
            static consteval std::size_t ComponentCount() noexcept
            {
                if constexpr(StoresDiagonal)
                {
                    return static_cast<std::size_t>(N) * (N + 1) / 2;
                }
                else
                {
                    return static_cast<std::size_t>(N) * (N - 1) / 2;
                }
            }

            template<int N>
            [[nodiscard]]
            static constexpr std::size_t IndependentIndex(const int mu_in, const int nu_in) noexcept
            {
                const std::size_t mu {static_cast<std::size_t>(mu_in)};
                const std::size_t nu {static_cast<std::size_t>(nu_in)};
                if constexpr(StoresDiagonal)
                {
                    return nu + mu * N - mu * (mu + 1) / 2;
                }
                else
                {
                    return nu + mu * N - (mu + 2) * (mu + 1) / 2;
                }
            }

            template<int N>
            [[nodiscard]]
            static constexpr bool IsIndependentComponent(const int mu, const int nu) noexcept
            {
                const bool in_bounds {mu >= 0 and mu < N and nu >= 0 and nu < N};
                if constexpr(StoresDiagonal)
                {
                    return in_bounds and mu <= nu;
                }
                else
                {
                    return in_bounds and mu < nu;
                }
            }

            template<int N>
            [[nodiscard]]
            static constexpr ComponentMap Locate(const int mu, const int nu) noexcept
            {
                if constexpr(not StoresDiagonal)
                {
                    if (mu == nu)
                    {
                        return {0, ComponentAccess::ImplicitZero};
                    }
                }

                if (mu <= nu)
                {
                    return {IndependentIndex<N>(mu, nu), ComponentAccess::Direct};
                }
                return {IndependentIndex<N>(nu, mu), ComponentAccess::Indirect};
            }
        };
    } // namespace Detail


    // T_{mu nu} = T_{nu mu}
    struct Symmetric : Detail::UpperTriangularSymmetryRelation<true>
    {
        template<typename T>
        [[nodiscard]]
        static T Read(const Detail::ComponentMap& component_map, const T* components) noexcept
        {
            return components[component_map.component];
        }

        template<typename T>
        static void Write(const Detail::ComponentMap& component_map, T* components, const T& value) noexcept
        {
            components[component_map.component] = value;
        }
    };

    // T_{mu nu} = -T_{nu mu}
    struct Antisymmetric : Detail::UpperTriangularSymmetryRelation<false>
    {
        template<typename T>
        [[nodiscard]]
        static T Read(const Detail::ComponentMap& component_map, const T* components) noexcept
        {
            if (component_map.access == Detail::ComponentAccess::ImplicitZero)
            {
                return Detail::Zero<T>();
            }
            const T& stored {components[component_map.component]};
            return component_map.access == Detail::ComponentAccess::Indirect ? -stored : stored;
        }

        template<typename T>
        static void Write(const Detail::ComponentMap& component_map, T* components, const T& value) noexcept
        {
            if (component_map.access != Detail::ComponentAccess::ImplicitZero)
            {
                components[component_map.component] = component_map.access == Detail::ComponentAccess::Indirect ? -value : value;
            }
        }
    };

    // T_{mu nu} = T_{nu mu}^\dagger
    struct AdjointSymmetric : Detail::UpperTriangularSymmetryRelation<true>
    {
        template<typename T>
        [[nodiscard]]
        static T Read(const Detail::ComponentMap& component_map, const T* components) noexcept
        {
            const T& stored {components[component_map.component]};
            return component_map.access == Detail::ComponentAccess::Indirect ? Detail::Adjoint(stored) : stored;
        }

        // Diagonal writes assume that value is Hermitian
        template<typename T>
        static void Write(const Detail::ComponentMap& component_map, T* components, const T& value) noexcept
        {
            components[component_map.component] = component_map.access == Detail::ComponentAccess::Indirect ? Detail::Adjoint(value) : value;
        }
    };

    // T_{mu nu} = -T_{nu mu}^\dagger
    struct AdjointAntisymmetric : Detail::UpperTriangularSymmetryRelation<true>
    {
        template<typename T>
        [[nodiscard]]
        static T Read(const Detail::ComponentMap& component_map, const T* components) noexcept
        {
            const T& stored {components[component_map.component]};
            return component_map.access == Detail::ComponentAccess::Indirect ? -Detail::Adjoint(stored) : stored;
        }

        // Diagonal writes assume that value is anti-Hermitian
        template<typename T>
        static void Write(const Detail::ComponentMap& component_map, T* components, const T& value) noexcept
        {
            components[component_map.component] = component_map.access == Detail::ComponentAccess::Indirect ? -Detail::Adjoint(value) : value;
        }
    };

    // T_{mu nu} = T_{nu mu}^\dagger, and T_{mu mu} = 0
    struct ZeroDiagonalAdjointSymmetric : Detail::UpperTriangularSymmetryRelation<false>
    {
        template<typename T>
        [[nodiscard]]
        static T Read(const Detail::ComponentMap& component_map, const T* components) noexcept
        {
            if (component_map.access == Detail::ComponentAccess::ImplicitZero)
            {
                return Detail::Zero<T>();
            }
            const T& stored {components[component_map.component]};
            return component_map.access == Detail::ComponentAccess::Indirect ? Detail::Adjoint(stored) : stored;
        }

        template<typename T>
        static void Write(const Detail::ComponentMap& component_map, T* components, const T& value) noexcept
        {
            if (component_map.access != Detail::ComponentAccess::ImplicitZero)
            {
                components[component_map.component] = component_map.access == Detail::ComponentAccess::Indirect ? Detail::Adjoint(value) : value;
            }
        }
    };

    template<typename PolicyT, typename T, int N>
    concept ComponentRelationPolicy = requires(const Detail::ComponentMap& component_map, const T* readable_components, T* writable_components, const T& value, const int mu, const int nu)
    {
        {PolicyT::template ComponentCount<N>()}                                 -> std::convertible_to<std::size_t>;
        {PolicyT::template Locate<N>(mu, nu)}                                   -> std::same_as<Detail::ComponentMap>;
        {PolicyT::template Read<T>(component_map, readable_components)}         -> std::same_as<T>;
        {PolicyT::template Write<T>(component_map, writable_components, value)} -> std::same_as<void>;
    };
} // namespace TensorSymmetry

#endif // LETTUCE_TENSOR_SYMMETRY_HPP
