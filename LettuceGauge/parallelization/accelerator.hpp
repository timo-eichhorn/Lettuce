#ifndef LETTUCE_ACCELERATOR_HPP
#define LETTUCE_ACCELERATOR_HPP

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
#include <cstdint>

// Try to make the code compatible with both CPU and GPU computations, i.e., standard C++ and CUDA/HIP using the utility macros/functions defined here
// Unfortunately, there is probably no way to avoid these macros without having to write much more code by hand

// Use macros to define CUDA/HIP keywords if compiling for GPU, otherwise define as expression that does nothing
// TODO: For now use __CUDACC__, but in the future it might be better to introduce a separate flag LETTUCE_GPU compatible with CUDA and HIP
// #if defined(LETTUCE_GPU)
#if defined(__CUDACC__)
#define lttc_host   __host__
#define lttc_device __device__
#define lttc_global __global__
#else
// #define lttc_host   ((void)0);
// #define lttc_device ((void)0);
// #define lttc_global ((void)0);
#define lttc_host
#define lttc_device
#define lttc_global
#endif

//-----
// The only way to launch device functions from the host in CUDA is through global functions/kernels which are called using the triple chevron syntax
// Therefore, we would like to wrap these calls 

// This probably needs to be a macro instead of a function template
// #define LaunchKernel()
// template<typename funcT>
// void LaunchKernel(funcT&& kernel)
// {
//     // Now if we use CUDA/HIP, we can launch __lttc_device__ functions from here
//     // Thus in practice, we pass the device function as argument to the LaunchKernel function
//     LaunchGPUKernel<<<n_blocks, n_threads_per_block>>>(/*placeholder*/);
//     // On the other hand, when using OpenMP, we can pass a regular function
//     LaunchCPUKernel(/*placeholder*/);
// }

// template<typename funcT>
// lttc_global
// void LaunchGPUKernel(funcT&& kernel)
// {
//     //...
// }

#define LaunchParallelFor() \
{                           \
    ApplyOperation<<<n_blocks, n_threads>>>();       \
}

template<typename FuncT>
lttc_global
void ApplyFunction(FuncT func, std::uint64_t n_max)
{
    std::uint64_t ind = blockDim.x * blockIdx.x + threadIdx.x;
    if (ind < n_max)
    {
        func(ind);
    }
}

// LaunchKernel() -> ApplyOperation()

// We have a function orelax() which applies to a single site
// This function is passed to something which handles the task distribution/parallelization
// This something needs to be a macro to be able to support both OpenMP and CUDA (?)
// Run(OrelaxFunctor)

#endif // LETTUCE_ACCELERATOR_HPP
