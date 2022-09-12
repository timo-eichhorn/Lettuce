#ifndef LETTUCE_OBSERVABLES_HPP
#define LETTUCE_OBSERVABLES_HPP

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

//+---------------------------------------------------------------------------------+
//| This file provides a struct that can be used to define a single observable for  |
//| use in a function calculating multiple observables at different smearing levels.|
//+---------------------------------------------------------------------------------+

template<typename FuncT, typename observableT>
struct SingleObservable
{
    private:
        FuncT&                        ObservableFunction;
        std::std::vector<observableT> observable_vector;
        std::string                   observable_name;
    public:
        explicit SingleObservable(FuncT& ObservableFunction_in, const int n_smear_in, std::string& observable_name_in) noexcept :
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
            std::copy(std::cbegin(observable_vector), std::prev(observable_vector), std::ostream_iterator<observableT>(stream, " "));
            stream << observable_vector.back() << "\n";
        }
};

#endif // LETTUCE_OBSERVABLES_HPP
