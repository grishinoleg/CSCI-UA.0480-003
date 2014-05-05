/* Oleg Grishin, NYU, 2014
  Parallel Computing, CSCI-UA.0480-003
*/

#include <stdio.h>
#include <stdlib.h>

#define SIZE 8192

/* Global variables */
int *a; /* Global array */

/* Function declarations */
void fill_array(); /* Fills array with random ints */
int find_max(); /* Finds max element of an array */

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
int find_max()
{
  int i, l_max = a[0];
  for (i = 1; i < SIZE; i++)
  {
    if (a[i] > l_max)
    {
      l_max = a[i];
    }
  }
  return l_max;
}

int main(int argc, char *argv[])
{
  a = (int *) malloc(SIZE * sizeof(int));

  fill_array();

  int max = find_max();

  printf("%d\n", max);

  return 0;
}
