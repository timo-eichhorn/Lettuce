#ifndef LETTUCE_COMPOSITE_ACTION_HPP
#define LETTUCE_COMPOSITE_ACTION_HPP

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

// TODO: Still heavily WIP, not included anywhere yet

// For std::integral_constant
#include <type_traits>

template<int First, int Last, typename funcT>
void for_constexpr(funcT&& function)
{
    if constexpr(First < Last)
    {
        f(std::integral_constant<int, First>{});
        for_constexpr<First + 1, Last>(std::forward(f));
    }
}

// For std::make_integer_sequence/std::integer_sequence
// Not available with CUDA yet?
#include <utility>

template <typename T, T... S, typename F>
constexpr void for_sequence(std::integer_sequence<T, S...>, F f)
{
    (static_cast<void>(f(std::integral_constant<T, S>{})), ...);
}

template<auto n, typename F>
constexpr void for_sequence(F f)
{
    for_sequence(std::make_integer_sequence<decltype(n), n>{}, f);
}

template<typename... ActionT>
class CompositeAction
{
    // Following components:
    // i) A number of actions (e.g. a gauge action and a fermionic action, or a gauge action and an additional Metadynamics action)
    // ii) Specify at which level the actions are supposed to be integrated? Not sure if that belongs here, since this is only required for molecular-dynamics-based updates
    public:
        std::tuple<ActionT&...> actions;
        constexpr std::size_t n_actions {sizeof...(ActionT)};

        CompositeAction(ActionT&&... Actions) noexcept :
        actions(std::forward<ActionT>(Actions)...)
        {}
};

// TODO: Move this to a different file later on

template<typename... IntegratorT>
class CompositeIntegrator
{
    public:
        std::tuple<IntegratorT...> integrators;
        constexpr std::size_t n_integrators {sizeof...(IntegratorT)};
    // TODO: Change name from HMC to something more generic (can also be SMD or Langevin...)
    template<typename UpdateFunctor>
    void operator()(UpdateFunctor& HMC, const double trajectory_length, const int n_step) const noexcept
    {
        // This would be for a single level
        for (int step_count = 0; step_count < n_step; ++step_count)
        {
            //
        }

        // This would be for two levels
        for (int timescale_count = 0; timescale_count < n_timescale; ++timescale_count)
        {
            for (int step_count = 0; step_count < n_step[timescale_count]; ++step_count)
            {
                //...
            }
        }
    }
};

// #include <iostream>
// #include <tuple>
// #include <type_traits>

// using namespace std;

// struct Act1
// {
//     void operator()(double a)
//     {
//         std::cout << "Act1: " << a << std::endl;
//     }
// };

// struct Act2
// {
//     void operator()(double a)
//     {
//         std::cout << "Act2: " << a << std::endl;
//     }
// };

// struct Act3
// {
//     void operator()(double a)
//     {
//         std::cout << "Act3: " << a << std::endl;
//     }
// };

// template<typename... ActionT>
// class CompositeAction
// {
//     public:
//         std::tuple<ActionT...> action_tuple;
//         CompositeAction(ActionT&&... Actions) noexcept :
//         action_tuple(std::forward<ActionT>(Actions)...)
//         {}
// };

// struct BaseAction
// {
//     void operator()(double a)
//     {
//         std::cout << "ActionBase: " << a << std::endl;
//     }
// };

// struct Action1 : BaseAction
// {
//     void operator()(double a)
//     {
//         std::cout << "Action1: " << a << std::endl;
//     }
// };

// struct Action2 : BaseAction
// {
//     void operator()(double a)
//     {
//         std::cout << "Action2: " << a << std::endl;
//     }
// };

// template<typename actT>
// void DoStuff(actT& act, double a)
// {
//     act(a);
// }

// int main()
// {
//     //Act1 a1;
//     //Act2 a2;
//     //CompositeAction act(std::forward<Act1>(a1), std::forward<Act2>(a2));
//     //std::get<0>(act.action_tuple)(3);
//     //cout << std::is_trivial<Act1>::value;
//     BaseAction a;
//     Action1 a1;
//     Action2 a2;
//     a(2);
//     a1(2);
//     a2(2);
//     DoStuff(a, 2);
//     DoStuff(a1, 2);
//     DoStuff(a2, 2);
//     return 0;
// }

#endif // LETTUCE_COMPOSITE_ACTION_HPP
