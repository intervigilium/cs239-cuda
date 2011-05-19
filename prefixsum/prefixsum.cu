/*
 * UCLA Spring 2011
 * CS239
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

#define NUM_BLOCKS 32
#define BLOCK_SIZE 512

__global__ void cuda_prefixsum(int *in_array, int *out_array, int size)
{
	// shared should be sized to blockDim.x
	extern __shared__ int shared[];

	unsigned int tid = threadIdx.x;
	unsigned int offset = 1;

	int i = tid;
	int j = tid + size / 2;
	int offset_i = CONFLICT_FREE_OFFSET(i);
	int offset_j = CONFLICT_FREE_OFFSET(j);
	shared[i + offset_i] = in_array[i];
	shared[j + offset_j] = in_array[j];

	// scan up
	for (int s = (size >> 1); s > 0; s >>= 1) {
		__syncthreads();

		if (tid < s) {
			int i = offset * (2 * tid + 1) - 1;
			int j = offset * (2 * tid + 2) - 1;
			i += CONFLICT_FREE_OFFSET(i);
			j += CONFLICT_FREE_OFFSET(j);
			shared[j] += shared[i];
		}
		offset <<= 1;
	}

	if (tid == 0) {
		shared[size - 1 + CONFLICT_FREE_OFFSET(size - 1)] = 0;
	}
	// scan down
	for (int s = 1; s < size; s <<= 1) {
		offset >>= 1;
		__syncthreads();

		if (tid < s) {
			int i = offset * (2 * tid + 1) - 1;
			int j = offset * (2 * tid + 2) - 1;
			i += CONFLICT_FREE_OFFSET(i);
			j += CONFLICT_FREE_OFFSET(j);
			int tmp = shared[i];
			shared[i] = shared[j];
			shared[j] += tmp;
		}
	}
	__syncthreads();
	// exclusive scan, need to shift all elements left, sum last one

	// copy data back to main memory
	// scan is exclusive, make it inclusive by left shifting elements
	if (tid > 0) {
		in_array[i - 1] = shared[i + offset_i];
	} else {
		// re-calc the last element, drop it in out array
		in_array[size - 1] +=
		    shared[size - 1 + CONFLICT_FREE_OFFSET(size - 1)];
		out_array[blockIdx.x] = in_array[size - 1];
	}
	in_array[j - 1] = shared[j + offset_j];
}

__global__ void cuda_updatesum(int *array, int *update_array, int size)
{
	extern __shared__ int shared[];

	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	int op = 0;

	if (blockIdx.x > 0) {
		op = update_array[blockIdx.x - 1];
	}

	shared[tid] = array[id] + op;
	array[id] = shared[tid];
}

void prefixsum(int blocks, int threads, int *array_h, int size)
{
	int *array_d;
	int *out_array_d;
	int *tmp_d;

	dim3 dim_grid(blocks, 1, 1);
	dim3 dim_block(threads, 1, 1);

	// allocate temp, block sum, and device arrays
	cudaMalloc((void **)&tmp_d, blocks * sizeof(int));
	cudaMalloc((void **)&out_array_d, blocks * sizeof(int));
	cudaMalloc((void **)&array_d, size * sizeof(int));
	cudaMemcpy(array_d, array_h, size * sizeof(int),
		   cudaMemcpyHostToDevice);

	// do prefix sum for each block
	cuda_prefixsum <<< dim_grid, dim_block,
	    threads * sizeof(int) >>> (array_d, out_array_d, size);
	// do prefix sum for block sum
	cuda_prefixsum <<< dim_grid, dim_block,
	    threads * sizeof(int) >>> (out_array_d, tmp_d, blocks);
	// update original array using block sum
	cuda_updatesum <<< dim_grid, dim_block,
	    threads * sizeof(int) >>> (array_d, out_array_d, size);

	// copy resulting array back to host
	cudaMemcpy(array_h, array_d, size * sizeof(int),
		   cudaMemcpyDeviceToHost);

	cudaFree(out_array_d);
	cudaFree(array_d);
}

void prefixsum_host(int *array_h, int size)
{
	for (int i = 0; i < size; i++) {
		if (i > 0) {
			array_h[i] += array_h[i - 1];
		}
	}
}

void usage(int which)
{
	switch (which) {
	default:
		printf("usage: prefixsum [-h|-b blocks|-t threads] max\n");
		break;
	case 1:
		printf("prefixsum requires numbers <= threads*blocks\n");
		break;
	}
}

void prepare_numbers(int **array, int count)
{
	int *numbers = (int *)malloc(count * sizeof(int));

	// load array
	for (int i = 0; i < count; i++) {
		numbers[i] = i;
	}

	*array = numbers;
}

void print_array(int *array, int count)
{
	for (int i = 0; i < count; i++) {
		printf("%d\n", array[i]);
	}
}

int main(int argc, char *argv[])
{
	int opt, host_mode, blocks, threads, max;
	int *array;

	// set options
	host_mode = 0;
	blocks = NUM_BLOCKS;
	threads = BLOCK_SIZE;
	while ((opt = getopt(argc, argv, "hb:t:")) != -1) {
		switch (opt) {
		case 'h':
			host_mode = 1;
			break;
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
		max = atoi(argv[optind]);
	} else {
		usage(0);
		return 0;
	}
	// pre-init numbers
	array = NULL;
	prepare_numbers(&array, max);

	if (host_mode) {
		printf("prefix sum using host\n");
		prefixsum_host(array, max);
	} else {
		printf("prefix sum using CUDA\n");
		prefixsum(blocks, threads, array, max);
	}

	// print array
	print_array(array, max);

	free(array);

	return 0;
}
