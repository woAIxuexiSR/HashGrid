#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/complex.h>

typedef unsigned int uint;
typedef unsigned short ushort;

#define BEM_NAMESPACE_BEGIN \
    namespace bem           \
    {
#define BEM_NAMESPACE_END }

BEM_NAMESPACE_BEGIN
#define PI 3.14159265358979323846f
using complex = thrust::complex<float>;
using randomState = curandState_t;
#define RAND_F (float)rand() / (float)RAND_MAX
#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define atomicAddCpxBlock(dst, value)                          \
    {                                                          \
        atomicAdd_block((float *)(dst), (value).real());       \
        atomicAdd_block(((float *)(dst)) + 1, (value).imag()); \
    }
#define atomicAddCpx(dst, value)                         \
    {                                                    \
        atomicAdd((float *)(dst), (value).real());       \
        atomicAdd(((float *)(dst)) + 1, (value).imag()); \
    }
/// Checks the result of a cuXXXXXX call and throws an error on failure
#define CU_CHECK_THROW(x)                                                                        \
    do                                                                                           \
    {                                                                                            \
        CUresult result = x;                                                                     \
        if (result != CUDA_SUCCESS)                                                              \
        {                                                                                        \
            const char *msg;                                                                     \
            cuGetErrorName(result, &msg);                                                        \
            throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + msg); \
        }                                                                                        \
    } while (0)

/// Checks the result of a cuXXXXXX call and prints an error on failure
#define CU_CHECK_PRINT(x)                                                            \
    do                                                                               \
    {                                                                                \
        CUresult result = x;                                                         \
        if (result != CUDA_SUCCESS)                                                  \
        {                                                                            \
            const char *msg;                                                         \
            cuGetErrorName(result, &msg);                                            \
            std::cout << FILE_LINE " " #x " failed with error " << msg << std::endl; \
        }                                                                            \
    } while (0)

/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define CUDA_CHECK_THROW(x)                                                                \
    do                                                                                     \
    {                                                                                      \
        cudaError_t result = x;                                                            \
        if (result != cudaSuccess)                                                         \
            throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + \
                                     cudaGetErrorString(result));                          \
    } while (0)

/// Checks the result of a cudaXXXXXX call and prints an error on failure
#define CUDA_CHECK_PRINT(x)                                                                                 \
    do                                                                                                      \
    {                                                                                                       \
        cudaError_t result = x;                                                                             \
        if (result != cudaSuccess)                                                                          \
            std::cout << FILE_LINE " " #x " failed with error " << cudaGetErrorString(result) << std::endl; \
    } while (0)

#define HOST_DEVICE __host__ __device__
constexpr uint32_t n_threads_linear = 256;

template <typename T>
HOST_DEVICE T div_round_up(T val, T divisor)
{
    return (val + divisor - 1) / divisor;
}

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements)
{
    return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}

template <typename K, typename T, typename... Types>
inline void linear_kernel(K kernel, T n_elements, Types... args)
{
    if (n_elements <= 0)
    {
        return;
    }
    kernel<<<n_blocks_linear(n_elements), n_threads_linear>>>(args...);
}

template <typename F>
__global__ void parallel_for_kernel(const size_t n_elements, F fun)
{
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    fun(i);
}

template <typename F>
inline void parallel_for(uint32_t shmem_size, size_t n_elements, F &&fun)
{
    if (n_elements <= 0)
    {
        return;
    }
    parallel_for_kernel<F><<<n_blocks_linear(n_elements), n_threads_linear, shmem_size>>>(n_elements, fun);
}

template <typename F>
inline void parallel_for(size_t n_elements, F &&fun)
{
    parallel_for(0, n_elements, std::forward<F>(fun));
}

template <typename F>
__global__ void parallel_for_block_kernel(F fun)
{
    fun(blockIdx.x, threadIdx.x);
}

template <typename F>
inline void parallel_for_block(uint32_t shmem_size, size_t n_blocks, size_t n_threads, F &&fun)
{
    if (n_blocks <= 0 || n_threads <= 0)
    {
        return;
    }
    parallel_for_block_kernel<F><<<n_blocks, n_threads, shmem_size>>>(fun);
}

template <typename F>
inline void parallel_for_block(size_t n_blocks, size_t n_threads, F &&fun)
{
    parallel_for_block(0, n_blocks, n_threads, std::forward<F>(fun));
}

inline __device__ void matmulABT(const float *A, const float *B, float *ABT)
{
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k)
            {
                sum += A[row * 3 + k] * B[col * 3 + k];
            }
            ABT[row * 3 + col] = sum;
        }
    }
}

BEM_NAMESPACE_END