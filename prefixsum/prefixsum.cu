/*
 * UCLA Spring 2011
 * CS239
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define NUM_BLOCKS 32
#define BLOCK_SIZE 512

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

void prefixsum(int blocks, int threads, int *array_h, int size)
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
		printf
		    ("usage: prefixsum [-h|-b blocks|-t threads] max\n");
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
	while ((opt = getopt(argc, argv, "h:b:t:")) != -1) {
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
    // need more blocks * threads than input numbers
	if (max >= blocks * threads) {
        usage(1);
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
