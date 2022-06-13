#ifndef LETTUCE_REDUCTIONS_HPP
#define LETTUCE_REDUCTIONS_HPP

// Non-standard library headers
#include "../defines.hpp"
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <vector>
//----------------------------------------
// Standard C headers
// ...

//----------------------------------------
// OpenmMP reductions

namespace reductions
{
    template <typename scalarT>
    scalarT Sum(const int n_thread)
    {
        std::vector<scalarT> partial_sums(n_thread);
        // Compute partial sums in separate threads
        // ...
        // Perform final reduction over partial sums to get total sum
        // ...
    }
}

#endif // LETTUCE_REDUCTIONS_HPP
