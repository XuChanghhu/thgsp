#include <torch/script.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif

int64_t cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}


static auto registry =
    torch::RegisterOperators().op("torch_gsp::cuda_version", &cuda_version);
