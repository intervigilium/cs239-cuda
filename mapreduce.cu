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
			shared[tid] = t_sum;
		}
		__syncthreads();
	}

	if (tid == 0) {
		out_array[blockIdx.x] = shared[0];
	}
}

int mapreduce(int blocks, int threads, int *array_h, int size)
{
	int res;
	int *array_d;
	int *in_ptr_d;
	int *out_array_d;
	int *tmp_d;

	dim3 dim_grid(blocks, 1, 1);
	dim3 dim_block(threads, 1, 1);

	cudaMalloc((void **)&array_d, size * sizeof(int));
	cudaMemcpy(array_d, array_h, size * sizeof(int),
		   cudaMemcpyHostToDevice);
	in_ptr_d = array_d;

	// do map
	map <<< dim_grid, dim_block, threads * sizeof(int) >>> (array_d, size);

	// do reduce in iterations
	for (unsigned int s = blocks / 2; s > 0; s >>= 1) {
		// allocate array equal to blocks/2
		cudaMalloc((void **)&out_array_d, s * sizeof(int));
		reduce <<< dim_grid, dim_block,
		    threads * sizeof(int) >>> (in_ptr_d, out_array_d, size);

		// free up the old input array
		tmp_d = in_ptr_d;
		// should check for tmp_d != array_h but we can just fail silently
		cudaFree(tmp_d);

		// set input array of next iteration to output array
		in_ptr_d = out_array_d;
		size = s;
	}

	// retrieve result
	cudaMemcpy(&res, out_array_d, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(out_array_d);
	cudaFree(array_d);

	return res;
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
	int *array;
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

	// check to make sure we are feeding in correct number of args
	if (argc == optind + 1) {
		filename = argv[optind];
	} else {
		usage(0);
		return 0;
	}

	// read file
	array = NULL;
	array_size = prepare_numbers(filename, &array);
	if (array_size < 0) {
		free(array);
		usage(1);
		return 0;
	} else if (array_size <= blocks * threads) {
		free(array);
		usage(2);
		return 0;
	}

	printf("mapreduce using CUDA\n");

	result = mapreduce(blocks, threads, array, array_size);

	printf("mapreduce result: %d\n", result);
	free(array);

	return 0;
}
