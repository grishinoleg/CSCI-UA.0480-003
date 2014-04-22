/* Oleg Grishin, NYU, 2014
  Parallel Computing, CSCI-UA.0480-003
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Global variables */
int num = 0; /* Num of cities */
int *d; /* Distances between cities */

/* Function declarations */
int get_distance(int i, int j); /* Gets distance between two cities */
void get_input(char filename[]); /* Reads input from file */

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

  rewind(fp);

  /* Number of elements below main diagonal (triangular number) */
  int elem_num = (num-1)*(num-2)/2;

  d = (int *) malloc(elem_num * sizeof(int));

  int dummy;

  /* Populate array of distances */
  for (int i = 0; i < num; i++)
  {
    for (int j = 0; j < num; j++)
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

int main(int argc, char *argv[])
{
  if(argc != 2)
  {
    printf("Usage: tsm filename\n");
    exit(1);
  }

  get_input(argv[1]);

  

  return 0;
}
