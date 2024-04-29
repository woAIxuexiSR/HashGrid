#pragma once
#include <common.h>
#include <cstddef>

BEM_NAMESPACE_BEGIN

template <typename T, size_t N>
struct PitchedPtr
{
        HOST_DEVICE PitchedPtr() : ptr(nullptr) {}

        template <typename... Sizes>
        HOST_DEVICE PitchedPtr(T *ptr, Sizes... sizes) : ptr(ptr)
        {
            set(ptr, sizes...);
        }

        template <typename... Sizes>
        HOST_DEVICE void set(T *ptr, Sizes... sizes)
        {
            static_assert(sizeof...(Sizes) == N, "Wrong number of sizes");
            size_t sizes_array[N] = {static_cast<size_t>(sizes)...};
            size[N - 1] = sizes_array[N - 1];
            stride[N - 1] = 1;
#pragma unroll
            for (int i = N - 2; i >= 0; --i)
            {
                size[i] = sizes_array[i];
                stride[i] = stride[i + 1] * size[i + 1];
            }
            this->ptr = ptr;
        }

        template <typename... Indices>
        HOST_DEVICE T &operator()(Indices... indices) const
        {
            static_assert(sizeof...(Indices) == N, "Wrong number of indices");
            return ptr[get_index(indices...)];
        }

        HOST_DEVICE T &operator()(int3 coord) const
        {
            static_assert(N == 3, "int3 operator can only be used with N=3");
            return ptr[get_index(coord.x, coord.y, coord.z)];
        }

        template <typename... Indices>
        HOST_DEVICE size_t get_index(Indices... indices) const
        {
            size_t indices_array[N] = {static_cast<size_t>(indices)...};
            size_t index = 0;
#pragma unroll
            for (int i = 0; i < N; ++i)
            {
                index += indices_array[i] * stride[i];
            }
            return index;
        }

        T *ptr;
        size_t stride[N];
        size_t size[N];
};

BEM_NAMESPACE_END