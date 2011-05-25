#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void map_kernel(int *g_data) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx] * g_data[idx];
}

__global__ void reduce_kernel(int *g_input, int *g_output) {
  extern __shared__ int s_data[]; // allocated at kernel launch

  // read input into shared memory
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[threadIdx.x] = g_input[idx];
  __syncthreads();

  // compute sum for the thread block
  for (int dist = blockDim.x / 2; dist > 0; dist /= 2) {
    if (threadIdx.x < dist)
      s_data[threadIdx.x] += s_data[threadIdx.x + dist];
    __syncthreads();
  }

  // write the block's sum to global memory
  if (threadIdx.x == 0)
    g_output[blockIdx.x] = s_data[0];
}

int main(int argc, char *argv[]) {
  // data set size in elements and bytes
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <num>\n", argv[1]);
    exit(1);
  }
  unsigned int n = atoi(argv[1]);
  unsigned int block_dim = 256;
  unsigned int nblocks = (n + block_dim - 1) / block_dim;

  unsigned int nbytes = block_dim * nblocks * sizeof(int);
  unsigned int smem_bytes = block_dim * sizeof(int);

  // allocate and initialize the data on the CPU
  int *h_a = (int *)malloc(nbytes);

  for (int i = 0; i < n; i++)
    h_a[i] = i + 1;

  // allocate memory on the GPU device
  int *d_a = 0, *d_out = 0;
  cudaMalloc((void **)&d_a, nbytes);
  cudaMalloc((void **)&d_out, nblocks * sizeof(int));

  // copy the input data from CPU to the GPU device
  cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);

  /*
   * <<< Dg, Db, Ns >>>
   * Dg :: dim3   dimension and size of grid
   * Db :: dim3   dimension and size of each block
   * Ns :: size_t number of bytes in shared memory that dynamically allocated
   *              per block
   */
  map_kernel<<<nblocks, block_dim>>>(d_a);
  // two stages of kernel execution
  reduce_kernel<<<nblocks, block_dim, smem_bytes>>>(d_a, d_out);
  reduce_kernel<<<1, nblocks, nblocks * sizeof(int)>>>(d_out, d_out);

  // copy the output from GPU device to CPU and print
  cudaMemcpy(h_a, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", h_a[0]);

  // release resources
  cudaFree(d_a);
  cudaFree(d_out);
  free(h_a);

  return 0;
}
