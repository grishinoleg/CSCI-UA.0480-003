/* Oleg Grishin, NYU, 2014
  Parallel Computing, CSCI-UA.0480-003
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define SIZE 8192
#define THREADS 512
#define BLOCKS 16

/* Function declarations */
void fill_array(int* a); /* Fills array with random ints */
void find_max(int* a, int* max); /* Sets up cuda for kernel code execution */
__global__ void maxReduce(int* in); /* Kernel function */
__global__ void findMaxKernelAcrossBlocks(int* l_a); /* Kernel function
for finding max element when blocks are done executing their local maxima */

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

  maxReduce<<<BLOCKS, THREADS>>>(l_a);
  findMaxKernelAcrossBlocks<<<1, 1>>>(l_a);

  cudaMemcpy(max, l_a, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(a, l_a, size, cudaMemcpyDeviceToHost);

  cudaFree(l_a);

}

__global__ void maxReduce(int* in)
{
  int i, idx = blockIdx.x*blockDim.x + threadIdx.x, a, b;

  /* sequential addressing: split in half, */
  for(i=blockDim.x/2; i > 0; i>>=1)
  {
    if (threadIdx.x < i) {
      a = in[idx];
      b = in[idx + i];
      in[idx] = a > b ? a : b;
    }
    __syncthreads();
  }

}

__global__ void findMaxKernelAcrossBlocks(int* l_a)
{
  int block_width = SIZE/BLOCKS, i, elem, max = l_a[0];

  for (i = 1; i < BLOCKS; i++)
  {
    elem = l_a[i*block_width];
    if (max < elem)
    {
      max = elem;
    }
  }

  l_a[0] = max;

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
