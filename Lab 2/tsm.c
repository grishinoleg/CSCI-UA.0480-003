/* Oleg Grishin, NYU, 2014
  Parallel Computing, CSCI-UA.0480-003
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define VERBOSE 0

/* Global variables */
int num = 0; /* Num of cities */
int *d; /* Distances between cities */
int *b; /* Stores best result: [dist, city1, city2, ..., cityn] */

/* Function declarations */
int get_distance(int i, int j); /* Gets distance between two cities */
void get_input(char filename[]); /* Reads input from file */
void output(); /* Outputs best result */
void print_distances(); /* For checking distances */
void solve(); /* Solves tsm */

int get_distance(int i, int j)
{
  if (i < j)
  {
    return get_distance(j, i);
  }
  else if (i == j)
  {
    return 0;
  }

  /* n(n-1)/2 entries below diagonal + j offset for new row */
  return d[i*(i-1)/2 + j];
}

void get_input(char filename[])
{
  FILE * fp;

  fp = fopen(filename, "r");

  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

  /* Count lines in file for easy array allocation and file reading */

  int ch = 0;

  while(!feof(fp))
  {
    ch = fgetc(fp);
    if(ch == '\n')
    {
      num++;
    }
  }

  /* Best result */

  b = (int *) malloc(num * sizeof(int));
  b[0] = 0;

  rewind(fp);

  /* Number of elements below main diagonal (triangular number) */
  int elem_num = num*(num-1)/2;

  d = (int *) malloc(elem_num * sizeof(int));

  int dummy;

  /* Populate array of distances */
  int i,j;
  for (i = 0; i < num; i++)
  {
    for (j = 0; j < num; j++)
    {
      if (j < i)
      {
        fscanf(fp,"%d ", &d[i*(i-1)/2 + j]);
      }
      else
      {
        fscanf(fp,"%d ", &dummy);
      }
    }
  }

  fclose(fp);
}

void solve(int current, int *path, int num_visited, int distance, int did_visit)
{
  int my_path[num];
  memcpy(my_path, path, num_visited*sizeof(int));
  int i;

  #pragma omp critical
  {
    if (VERBOSE)
    {
      printf("Current: %d, visited: %d, distance: %d; thread: %d\nPath:", current, num_visited, distance, omp_get_thread_num());
      for(i = 0; i < num_visited; i++)
      {
        printf(" %d", my_path[i]);
      }
      printf("\n");
      printf("Current %d, num_visited %d, num %d\n", current, num_visited, num);
    }
  }

  if (num_visited <= num-2)
  {
    for(i = 1; i < num; i++)
    {
      if (!(did_visit & (1 << i)))
      {
        int new_dist = distance + get_distance(current, i);
        if ((b[0] && b[0] > new_dist) || !b[0])
        {
          my_path[num_visited] = i;
          solve(i, my_path, num_visited+1, new_dist, did_visit | (1 << i));
        }
      }
    }
  }
  else
  {
    if (VERBOSE)
    {
      printf("Reached final at current %d and distance %d for thread %d\n", current, distance, omp_get_thread_num());
    }
    #pragma omp critical
    {
      if (!b[0] || b[0] > distance)
      {
        b[0] = distance;
        memcpy(&b[1], &my_path, (num-1)*sizeof(int));
      }
    }
  }
}

void print_distances()
{
  int i, j;
  for (i=0; i < num; i++)
  {
    for (j=0; j<num; j++)
    {
      printf("%d ", get_distance(i,j));
    }
    printf("\n");
  }
  printf("\n\n");
}

void output()
{
  printf("Best path: 0");
  int i;
  for (i = 1; i < num; i++)
  {
    printf(" %d", b[i]);
  }
  printf("\nDistance: %d\n", b[0]);
}

int main(int argc, char *argv[])
{
  if(argc != 2)
  {
    printf("Usage: tsm filename\n");
    exit(1);
  }

  get_input(argv[1]);

  omp_set_num_threads(num-1);

  int i;

  #pragma omp parallel for
  for (i = 1; i < num; i++)
  {
    int path[num-1];
    path[0] = i;
    int did_visit = 0;
    did_visit |= (1 << i);

    solve(i, path, 1, get_distance(0, i), did_visit);
  }

  output();

  return 0;
}
