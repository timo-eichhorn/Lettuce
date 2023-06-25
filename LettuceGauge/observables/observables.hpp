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
        //       Perhaps there is an existing implementation for one of the CUDA/HIP libraries?
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

// TODO: CreateObservable doesn't work if the function is overloaded (like the topological charge)
// void Observables(const GaugeField& Gluon, GaugeField& Gluonchain, std::ofstream& wilsonlog, const int n_count, const int n_smear)
// {
//     // Action
//     // ActionImproved
//     // ActionUnnormalized
//     // WLoop2
//     // WLoop4
//     // WLoop8
//     // PLoopRe
//     // PLoopIm
//     // PLoop
//     // TopologicalChargeSymm
//     // TopologicalChargeUnimproved
//     auto Obs_WilsonAction   {CreateObservable<double>(WilsonAction::ActionNormalized, n_smear, "Wilson_Action")};
//     auto Obs_ImprovedAction {CreateObservable<double>(WilsonAction::ActionNormalized, n_smear, "Improved_Action")};
//     auto Obs_Plaquette      {CreateObservable<double>(PlaquetteSum, n_smear, "Plaquette")};
//     auto Obs_WLoop2         {CreateObservable<double>(WilsonLoop<0, 2, true>, n_smear, "Wilson_loop(L=2)")};
//     auto Obs_WLoop4         {CreateObservable<double>(WilsonLoop<2, 4, true>, n_smear, "Wilson_loop(L=4)")};
//     auto Obs_WLoop8         {CreateObservable<double>(WilsonLoop<4, 8, true>, n_smear, "Wilson_loop(L=8)")};
//     auto Obs_Ploop          {CreateObservable<std::complex<double>>(PolyakovLoop, n_smear, "Polyakov_loop")};
//     auto Obs_ClovCharge     {CreateObservable<double>(TopChargeGluonicSymm, n_smear, "TopChargeClov")};
//     auto Obs_PlaqCharge     {CreateObservable<double>(TopChargeGluonicUnimproved, n_smear, "TopChargePlaq")};

//     // Unsmeared
//     Obs_WilsonAction.Calculate(0, Gluon);
//     Obs_ImprovedAction.Calculate(0, Gluon);
//     Obs_Plaquette.Calculate(0, Gluon);
//     Obs_WLoop2.Calculate(0, Gluon, Gluonchain);
//     Obs_WLoop4.Calculate(0, Gluon, Gluonchain);
//     Obs_WLoop8.Calculate(0, Gluon, Gluonchain);
//     Obs_Ploop.Calculate(0, Gluon);
//     Obs_ClovCharge.Calculate(0, Gluon);
//     Obs_PlaqCharge.Calculate(0, Gluon);

//     // Begin smearing
//     if (n_smear > 0)
//     {
//         StoutSmearing4D(Gluon, Gluonsmeared1, rho_stout);
//         Obs_WilsonAction.Calculate(1, Gluonsmeared1);
//         Obs_ImprovedAction.Calculate(1, Gluonsmeared1);
//         Obs_Plaquette.Calculate(1, Gluonsmeared1);
//         Obs_WLoop2.Calculate(1, Gluonsmeared1, Gluonchain);
//         Obs_WLoop4.Calculate(1, Gluonsmeared1, Gluonchain);
//         Obs_WLoop8.Calculate(1, Gluonsmeared1, Gluonchain);
//         Obs_Ploop.Calculate(1, Gluonsmeared1);
//         Obs_ClovCharge.Calculate(1, Gluonsmeared1);
//         Obs_PlaqCharge.Calculate(1, Gluonsmeared1);
//     }

//     // Further smearing steps
//     for (int smear_count = 2; smear_count <= n_smear; ++smear_count)
//     {
//         // Even
//         if (smear_count % 2 == 0)
//         {
//             StoutSmearingN(Gluonsmeared1, Gluonsmeared2, n_smear_skip, rho_stout);
//             Obs_WilsonAction.Calculate(smear_count, Gluonsmeared2);
//             Obs_ImprovedAction.Calculate(smear_count, Gluonsmeared2);
//             Obs_Plaquette.Calculate(smear_count, Gluonsmeared2);
//             Obs_WLoop2.Calculate(smear_count, Gluonsmeared2, Gluonchain);
//             Obs_WLoop4.Calculate(smear_count, Gluonsmeared2, Gluonchain);
//             Obs_WLoop8.Calculate(smear_count, Gluonsmeared2, Gluonchain);
//             Obs_Ploop.Calculate(smear_count, Gluonsmeared2);
//             Obs_ClovCharge.Calculate(smear_count, Gluonsmeared2);
//             Obs_PlaqCharge.Calculate(smear_count, Gluonsmeared2);
//         }
//         // Odd
//         else
//         {
//             StoutSmearingN(Gluonsmeared2, Gluonsmeared1, n_smear_skip, rho_stout);
//             Obs_WilsonAction.Calculate(smear_count, Gluonsmeared1);
//             Obs_ImprovedAction.Calculate(smear_count, Gluonsmeared1);
//             Obs_Plaquette.Calculate(smear_count, Gluonsmeared1);
//             Obs_WLoop2.Calculate(smear_count, Gluonsmeared1, Gluonchain);
//             Obs_WLoop4.Calculate(smear_count, Gluonsmeared1, Gluonchain);
//             Obs_WLoop8.Calculate(smear_count, Gluonsmeared1, Gluonchain);
//             Obs_Ploop.Calculate(smear_count, Gluonsmeared1);
//             Obs_ClovCharge.Calculate(smear_count, Gluonsmeared1);
//             Obs_PlaqCharge.Calculate(smear_count, Gluonsmeared1);
//         }
//     }

//     // Save to files
//     // TODO: Rename to SaveToStream?
//     Obs_WilsonAction.SaveToFile(wilsonlog);
//     Obs_ImprovedAction.SaveToFile(wilsonlog);
//     Obs_Plaquette.SaveToFile(wilsonlog);
//     // Wilson_Action(unnormalized)
//     Obs_WLoop2.SaveToFile(wilsonlog);
//     Obs_WLoop4.SaveToFile(wilsonlog);
//     Obs_WLoop8.SaveToFile(wilsonlog);
//     // Obs_Ploop.SaveToFile(wilsonlog);
//     // Polyakov_loop(Re)
//     // Polyakov_loop(Im)
//     Obs_ClovCharge.SaveToFile(wilsonlog);
//     Obs_PlaqCharge.SaveToFile(wilsonlog);
// }

#endif // LETTUCE_OBSERVABLES_HPP
