#ifndef LETTUCE_OBSERVABLES_HPP
#define LETTUCE_OBSERVABLES_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
//----------------------------------------
// Standard C headers
// ...

//+---------------------------------------------------------------------------------+
//| This file provides a struct that can be used to define a single observable for  |
//| use in a function calculating multiple observables at different smearing levels.|
//| Since partial deduction for class templates is not possible, there is a helper  |
//| function to create the structs where the type of observableT is manually        |
//| specified, while the compiler handles the type of the function.                 |
//| Of course, the struct can also be directly constructed using decltyp(function)  |
//| as second template parameter.                                                   |
//+---------------------------------------------------------------------------------+

template<typename observableT, typename FuncT>
struct SingleObservable
{
    private:
        // We should pass a functor here, since for functors, the exact type is known at compile-time
        // For functions, only the signature is known in general
        FuncT&                   ObservableFunction;
        std::vector<observableT> observable_vector;
        std::string              observable_name;
    public:
        explicit SingleObservable(FuncT& ObservableFunction_in, const int n_smear_in, const std::string& observable_name_in) noexcept :
        ObservableFunction(ObservableFunction_in), observable_vector(n_smear_in), observable_name(observable_name_in)
        {}

        // TODO: Probably need a custom std::forward implementation if we want this to work with GPUs
        //       Perhaps there is an existinc implementation for one of the CUDA/HIP libraries?
        template<typename... ParamsT>
        void Calculate(const int smearing_level, ParamsT&&... params) noexcept
        {
            observable_vector[smearing_level] = ObservableFunction(std::forward<ParamsT>(params)...);
        }

        observableT operator[](const int i) const noexcept
        {
            return observable_vector[i];
        } 

        void SaveToFile(std::ofstream& stream) const noexcept
        {
            stream << observable_name << ": ";
            std::copy(std::cbegin(observable_vector), std::prev(std::cend(observable_vector)), std::ostream_iterator<observableT>(stream, " "));
            stream << observable_vector.back() << "\n";
        }
};

// Since partial deduction for class templates is not possible, we use this helper function to manually specify the type of observableT, while the compiler
// handles the type of FuncT automatically. Of course, we could also directly construct SingleObservable using decltyp(function) as second template parameter.

template<typename observableT, typename FuncT>
[[nodiscard]]
auto CreateObservable(FuncT& ObservableFunction, const int n_smear_in, const std::string& observable_name) noexcept
{
    return SingleObservable<observableT, FuncT>{ObservableFunction, n_smear_in, observable_name};
}

// All observables:
// Smoothing method (cooling, smearing, gradient flow, ...)
// Number of smoothing steps
// Smoothing skip steps
// Observables (variadic template parameter?)

// template<typename observableT>
// struct SingleObservable
// {
//     private:
//         std::vector<observableT> observable_vector;
//         std::string              observable_name;
//     public:
//         //...
//         template<typename FuncT>
//         struct impl
//         {
//             FuncT& ObservableFunction;

//             impl(FuncT& ObservableFunction_in) noexcept :
//             ObservableFunction(ObservableFunction_in)
//             {}
//         };
//         // Deduction guide
//         template<typename FuncT>
//         impl(FuncT& ObservableFunction_in) -> impl<FuncT>;

//         SingleObservable() noexcept :
//         ()
//         {}

//         template<typename... ParamsT>
//         void Calculate(const int smearing_level, ParamsT&&... params) noexcept
//         {
//             observable_vector[smearing_level] = impl.ObservableFunction(std::forward<ParamsT>(params)...);
//         }

//         double operator[](const int i) const noexcept
//         {
//             return observable_vector[i];
//         } 

//         void SaveToFile(std::ofstream& stream) const noexcept
//         {
//             stream << observable_name << ": ";
//             std::copy(std::cbegin(observable_vector), std::prev(std::cend(observable_vector)), std::ostream_iterator<double>(stream, " "));
//             stream << observable_vector.back() << "\n";
//         }
// };

#endif // LETTUCE_OBSERVABLES_HPP
