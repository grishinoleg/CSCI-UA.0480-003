/* Oleg Grishin, NYU, 2014
  Parallel Computing, CSCI-UA.0480-003
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define SIZE 65536
#define THREADS 512
#define BLOCKS 128

/* Function declarations */
void fill_array(int* a); /* Fills array with random ints */
void find_max(int* a, int* max); /* Sets up cuda for kernel code execution */
__global__ void maxReduce(int* in, int isLast); /* Kernel function */

void fill_array(int* a)
{
  int i;
  srand(time(NULL));
  for (i = 0; i < SIZE; i++)
  {
    a[i] = rand() % 100000;
  }
}

void find_max(int* a, int* max)
{
  int size = SIZE * sizeof(int);
  int *l_a;
  cudaMalloc((void**)&l_a, size);
  cudaMemcpy(l_a, a, size, cudaMemcpyHostToDevice);

  maxReduce<<<BLOCKS, THREADS, THREADS*sizeof(int)>>>(l_a, 0);
  maxReduce<<<1, BLOCKS, BLOCKS*sizeof(int)>>>(l_a, 1);

  cudaMemcpy(max, l_a, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(a, l_a, size, cudaMemcpyDeviceToHost);

  cudaFree(l_a);

}

__global__ void maxReduce(int* in, int isLast)
{
  extern __shared__ int in_s[];
  int i, tid = threadIdx.x, idx, a, b, upper_lim;

  if (isLast)
  {
    idx = threadIdx.x*THREADS;
    upper_lim = THREADS/2;
  }
  else
  {
    idx = blockIdx.x*blockDim.x + threadIdx.x;
    upper_lim = blockDim.x/2;
  }

  in_s[tid] = in[idx];
  __syncthreads();

  /* sequential addressing: split in half, */
  for(i=upper_lim; i > 0; i>>=1)
  {
    if (tid < i) {
      a = in_s[tid];
      b = in_s[tid + i];
      in_s[tid] = a > b ? a : b;
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    in[blockIdx.x*blockDim.x] = in_s[0];
  }

}

int main(int argc, char *argv[])
{
  int* a = (int *) malloc(SIZE * sizeof(int));

  fill_array(a);

  int max;

  find_max(a, &max);

  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(1);
  }

  printf("%d\n", max);

  exit(0);
}
