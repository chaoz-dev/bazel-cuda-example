#include <cuda_runtime.h>
#include <iostream>

int main()
{
    // Simple example program to show all CUDA devices and their properties.
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
