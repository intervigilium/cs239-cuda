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

__global__ void map(int *array, int size)
{
	// gridDim.x * blockDim.x >= size / 2
	// shared should be sized to blockDim.x * 2
	extern __shared__ int shared[];

	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// do map in shared mem
	if (id < size) {
		// do first operation
		shared[tid] = array[id];
		array[id] = rand(shared[tid], shared[tid]);
	}
	if (id + blockDim.x < size) {
		// do second operation
		shared[tid + blockDim.x] = array[id + blockDim.x];
		array[id + blockDim.x] =
		    rand(shared[tid + blockDim.x], shared[tid + blockDim.x]);
	}
}

__global__ void reduce(int *in_array, int *out_array, int size)
{
	// gridDim.x * blockDim.x >= size / 2
	// shared should be sized to blockDim.x
	extern __shared__ int shared[];

	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// first sum
	int t_sum = (id < size) ? in_array[id] : 0;
	if (id + blockDim.x < size) {
		// current reduce function is sum
		t_sum = sum(in_array[id + blockDim.x], t_sum);
	}

	shared[tid] = t_sum;
	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
            // current reduce function is sum
            t_sum = sum(t_sum, shared[tid + s]);
            shared[tid = t_sum];
		}
		__syncthreads();
	}

	if (tid == 0) {
		out_array[blockIdx.x] = shared[0];
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
	map <<< dim_grid, dim_block, threads * sizeof(int) >>> (array_d,
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
