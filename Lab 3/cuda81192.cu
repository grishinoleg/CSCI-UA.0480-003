/* Oleg Grishin, NYU, 2014
  Parallel Computing, CSCI-UA.0480-003
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// #define SIZE 8192
// #define THREADS 256
#define SIZE 16
#define THREADS 4

/* Global variables */
int *a; /* Global array */
int max_num;

/* Function declarations */
void fill_array(); /* Fills array with random ints */
void find_max(int* a, int* max); /* Sets up cuda for kernel code execution */
__global__ void findMaxKernel(int* l_a, int* l_max); /* Kernel function */
void print_array(int* a, int n); /*  Prints an array */

void fill_array()
{
  int i;
  for (i = 0; i < SIZE; i++)
  {
    a[i] = rand() % 100000;
  }
}

/* Finding max can be done in fill_array, but since
  the program is going to be parallelized, it is better to use
  two functions for demonstrativeness */
void find_max(int* a, int* max)
{
  int size = SIZE * sizeof(int);
  int *l_a, *l_max;
  cudaMalloc((void**)&l_a, size);
  cudaMemcpy(l_a, a, size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&l_max, sizeof(int));
  int first_max = a[0];
  cudaMemcpy(l_max, &first_max, sizeof(int), cudaMemcpyHostToDevice);

  findMaxKernel<<<ceil(SIZE/THREADS), THREADS>>>(l_a, l_max);

  cudaFree(l_a);

  cudaMemcpy(max, l_max, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(&l_max);

}

__global__ void findMaxKernel(int* l_a, int* l_max)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < SIZE && *l_max < l_a[i])
  {
    *l_max = l_a[i];
  }
}

void print_array(int* a, int n)
{
  int i;
  for (i = 0; i < n; i++)
  {
    printf("%d ", a[i]);
  }
  printf("\n");
}

int main(int argc, char *argv[])
{
  a = (int *) malloc(SIZE * sizeof(int));

  fill_array();

  // print_array(a, SIZE);

  find_max(a, &max_num);

  printf("%d\n", max_num);

  return 0;
}
