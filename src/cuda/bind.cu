#include "common.h"
#include "gpu_memory.h"
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <curand_kernel.h>

using namespace bem;

torch::Tensor knn(torch::Tensor xi_, torch::Tensor xf_, int k)
{
    int batch_size = xi_.size(0);
    int n_dim = xi_.size(1);
    PitchedPtr<long, 2> xi((long *)xi_.data_ptr(), batch_size, n_dim);
    PitchedPtr<float, 2> xf((float *)xf_.data_ptr(), batch_size, n_dim);
    torch::Tensor neigs_ = torch::zeros({batch_size, k, n_dim}, torch::dtype(torch::kLong).device(torch::kCUDA));
    PitchedPtr<long, 3> neigs((long *)neigs_.data_ptr(), batch_size, k, n_dim);
    torch::Tensor rs_ = torch::rand({batch_size, k, n_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    PitchedPtr<float, 3> rs((float *)rs_.data_ptr(), batch_size, k, n_dim);

    parallel_for(batch_size, [=] __device__(int i) {
        for (int j = 0; j < k; j++)
            for (int d = 0; d < n_dim; d++)
            {
                float r = rs(i, j, d);
                long x_int = xi(i, d);
                float x_intf = xf(i, d);
                if (r < x_intf)
                    neigs(i, j, d) = x_int + 1;

                else
                    neigs(i, j, d) = x_int;
            }
    });

    return neigs_;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("knn", &knn, "k-nearest neighbors");
}
