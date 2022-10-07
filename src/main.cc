#include <cuda_runtime.h>
#include <iostream>

#define CUDA_ASSERT(err_n) cuda_assert(err_n, true, __FILE__, __LINE__);
inline void cuda_assert(cudaError_t err_n, bool terminate, const char* filename, int lineno)
{
    if (err_n == cudaSuccess)
    {
        return;
    }

    std::cerr << filename << ": " << lineno << std::endl
              << cudaGetErrorName(err_n) << ": " << std::endl
              << cudaGetErrorString(err_n) << std::endl;

    if (terminate)
    {
        std::exit(EXIT_FAILURE);
    }
}

int main()
{
    // Show CUDA version.
    int driver_version  = -1;
    int runtime_version = -1;

    CUDA_ASSERT(cudaDriverGetVersion(&driver_version));
    CUDA_ASSERT(cudaRuntimeGetVersion(&runtime_version));

    std::cout << "CUDA driver version: " << driver_version << std::endl;
    std::cout << "CUDA runtime version: " << runtime_version << std::endl;

    // Show all CUDA devices and their properties.
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);

    std::cout << "Found " << num_devices << " CUDA device(s)." << std::endl;
    for (int i = 0; i < num_devices; ++i)
    {
        cudaDeviceProp device;
        cudaGetDeviceProperties(&device, i);

        std::cout << "Device number: " << i << std::endl;
        std::cout << "  Device name: " << device.name << std::endl;
        std::cout << "  Compute capability: " << device.major << "." << device.minor << std::endl;
        std::cout << "  Device clock rate (MHz): " << device.clockRate / 1.0e3 << std::endl;
        std::cout << "  Device memory (GB): " << device.totalGlobalMem / 1.0e9 << std::endl;
        std::cout << "  Memory clock rate (MHz): " << (device.memoryClockRate / 2) / 1.0e3 << std::endl;
        std::cout << "  Memory bus width (bits): " << device.memoryBusWidth << std::endl;
        std::cout << "  Memory bandwidth (GB/s): " << 2.0 * device.memoryClockRate * (device.memoryBusWidth / 8) / 1.0e6
                  << std::endl;
    }

    return 0;
}
