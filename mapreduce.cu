/*
 * UCLA Spring 2011
 * CS239
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define NUM_BLOCKS 8
#define BLOCK_SIZE 512

// define map/reduce function type
typedef int (*map_function_t) (int, int);
typedef int (*reduce_function_t) (int, int);

__device__ int rand(int init0, int init1)
{
	// multiply-with-carry RNG
	init0 = 36969 * (init0 & 65535) + (init0 >> 16);
	init1 = 18000 * (init1 & 65535) + (init1 >> 16);
	return (init0 << 16) + init1;	/* 32-bit result */
}

__device__ int fma0(int op0, int op1)
{
	return op0 + op0 * op1;
}

__device__ int sum(int op0, int op1)
{
	return op0 + op1;
}

// NVIDIA reference implementation
template < unsigned int blockSize >
    __global__ void reduce(int *g_idata, int *g_odata, unsigned int n)
{
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;

	while (i < n) {
		sdata[tid] += g_idata[i] + g_idata[i + blockSize];
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] += sdata[tid + 64];
		}
		__syncthreads();
	}

	if (tid < 32) {
		if (blockSize >= 64)
			sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32)
			sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16)
			sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8)
			sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4)
			sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2)
			sdata[tid] += sdata[tid + 1];
	}

	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
	}
}

__global__ void mapreduce(int *array, int count, int *g_cache, int *result)
{
	// gridDim.x is number of blocks
	// blockDim.x is number of threads per block
	// b_cache should be equally sized to blockDim.x
	// g_cache should be equally sized to gridDim.x
	extern __shared__ int b_cache[];

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i, thread_work_size, thread_offset;

	if (tid == gridDim.x * blockDim.x - 1) {
		int normal_work_size = count / (gridDim.x * blockDim.x);
		thread_offset = tid * normal_work_size;
		thread_work_size = count - thread_offset;
	} else {
		thread_work_size = count / (gridDim.x * blockDim.x);
		if (thread_work_size < 1) {
			thread_work_size = 1;
		}
		thread_offset = tid * thread_work_size;
	}

	// map section
	b_cache[threadIdx.x] = array[thread_offset];
	for (i = thread_offset + 1; i < thread_offset + thread_work_size; i++) {
		b_cache[threadIdx.x] = rand(b_cache[threadIdx.x], array[i]);
	}
	__syncthreads();

	// reduce section

	// get the largest number in a block
	if (threadIdx.x == 0) {
		int largest = b_cache[0];
		for (i = 1; i < blockDim.x; i++) {
			if (b_cache[i] > largest) {
				largest = b_cache[i];
			}
		}
		g_cache[blockIdx.x] = largest;
	}
	__syncthreads();

	// get the largest number in all blocks
	if (tid == 0) {
		int largest = g_cache[0];
		for (i = 1; i < gridDim.x; i++) {
			if (g_cache[i] > largest) {
				largest = g_cache[i];
			}
		}
		*result = largest;
	}
}

void usage(int which)
{
	switch (which) {
	default:
		printf("usage: mapreduce [-b blocks|-t threads] <filename>\n");
		break;
	case 1:
		printf("mapreduce input format:\nnum count\n1\n...\nn n\n");
		break;
	case 2:
		printf("mapreduce requires numbers >= threads*blocks\n");
		break;
	}
}

int prepare_numbers(const char *filename, int **array)
{
	int count, input, i;
	FILE *file;

	file = fopen(filename, "r");

	// count of data is first line
	fscanf(file, "%d", &count);
	int *numbers = (int *)malloc(count * sizeof(int));

	// load array
	for (i = 0; i < count; i++) {
		if (fscanf(file, "%d", &input) < 0) {
			break;
		}
		numbers[i] = input;
	}
	fclose(file);

	if (count != i) {
		free(numbers);
		return -1;
	} else {
		*array = numbers;
		return count;
	}
}

int main(int argc, char *argv[])
{
	int opt, blocks, threads, array_size, result;
	int *array_h, *array_d, *result_d, *cache_d;
	char *filename;

	// set options
	blocks = NUM_BLOCKS;
	threads = BLOCK_SIZE;
	while ((opt = getopt(argc, argv, "b:t:")) != -1) {
		switch (opt) {
		case 'b':
			blocks = atoi(optarg);
			break;
		case 't':
			threads = atoi(optarg);
			break;
		default:
			usage(0);
			return 0;
		}
	}
	dim3 dim_grid(blocks, 1, 1);
	dim3 dim_block(threads, 1, 1);

	// check to make sure we are feeding in correct number of args
	if (argc == optind + 1) {
		filename = argv[optind];
	} else {
		usage(0);
		return 0;
	}

	// read file
	array_h = NULL;
	array_size = prepare_numbers(filename, &array_h);
	if (array_size < 0) {
		free(array_h);
		usage(1);
		return 0;
	} else if (array_size <= blocks * threads) {
		free(array_h);
		usage(2);
		return 0;
	}

	result = 0;
	printf("mapreduce using CUDA\n");
	// move to device
	cudaMalloc((void **)&array_d, array_size * sizeof(int));
	cudaMemcpy(array_d, array_h, array_size * sizeof(int),
		   cudaMemcpyHostToDevice);

	// allocate device only structures
	cudaMalloc((void **)&result_d, sizeof(int));
	cudaMalloc((void **)&cache_d, blocks * sizeof(int));

	// run kernel
	mapreduce <<< dim_grid, dim_block, threads * sizeof(int) >>> (array_d,
								      array_size,
								      cache_d,
								      result_d);

	// retrieve result
	cudaMemcpy(&result, result_d, sizeof(int), cudaMemcpyDeviceToHost);

	// cleanup
	cudaFree(array_d);
	cudaFree(result_d);
	cudaFree(cache_d);

	printf("mapreduce result: %d\n", result);
	free(array_h);

	return 0;
}
