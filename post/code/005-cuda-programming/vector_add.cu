#include <iostream>
#include <cuda_runtime.h>

// __global__ indicates the add function is a kernel that runs on the GPU
__global__ void add(int *a, int *b, int *c, int size)
{
    // built-in variables define the index of the block and thread. 
    // Cuda organizesd execution in grids of blocks, and each block contains multiple threads.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        c[tid] = a[tid] + b[tid];
    }
}

#define cudaCheckError() {                                                    \
    cudaError_t e = cudaGetLastError();                                       \
    if (e != cudaSuccess) {                                                   \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
}

int main()
{
    const int size = 5;

    int h_a[size] = {1, 2, 3, 4, 5};
    int h_b[size] = {10, 20, 30, 40, 50};
    int h_c[size] = {0};

    int *d_a, *d_b, *d_c;
    size_t bytes = size * sizeof(int);

    // Allocate memory on the GPU
    cudaMalloc(&d_a, bytes);
    cudaCheckError();
    cudaMalloc(&d_b, bytes);
    cudaCheckError();
    cudaMalloc(&d_c, bytes);
    cudaCheckError();

    // Copy data from host to device (CPU -> GPU)
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaCheckError();

    // Launch kernel with 1 block of 5 threads
    add<<<1, size>>>(d_a, d_b, d_c, size);
    cudaCheckError();

    // Copy the result back to the host (GPU -> CPU)
    
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Print the result
    std::cout << "Result: ";
    for (int i = 0; i < size; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}