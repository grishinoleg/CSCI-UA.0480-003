#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// Defines for more readable code

#define MASTER 0
#define IS_MASTER my_rank == MASTER


/***** Globals ******/
float **a; /* The coefficients */
float *a_row; /* The coefficients of one row of a */
float *x;  /* The unknowns */
float *x_old; /* Old value of unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */


/****** Function declarations */
void allocate_matrices_slave(); /* Allocate matrices for slave processes */
int check_error(); /* Check absolute relative error */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */
float get_new_x(int rank);

/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/* Since get_input is not called by non-master processes, we allocate
a, b and x */

void allocate_matrices_slave()
{
  /* Keeping double pointer for simplicity - no need to rewrite calculations
  for master process in get_new_x() function */

  a_row = (float *)malloc(num * sizeof(float));
  if( !a_row)
  {
    printf("Cannot allocate a!\n");
    exit(1);
  }

  x = (float *) malloc(num * sizeof(float));
  if( !x)
  {
    printf("Cannot allocate x!\n");
    exit(1);
  }


  b = (float *) malloc(num * sizeof(float));
  if( !b)
  {
    printf("Cannot allocate b!\n");
    exit(1);
  }
}

// Returns positive value if the relative absolute error is bigger than err

int check_error()
{
  int is_error = 0, i;
  for (i = 0; i < num; i++)
    if (fabs((x[i] - x_old[i]) / x[i]) > err)
      is_error = 1;
  return is_error;
}


/*
 Conditions for convergence (diagonal dominance):
 1. diagonal element >= sum of all other elements of the row
 2. At least one diagonal element > sum of all other elements of the row
*/
void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;

  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[i][i]);

    for(j = 0; j < num; j++)
       if( j != i)
   sum += fabs(a[i][j]);

    if( aii < sum)
    {
      printf("The matrix will not converge\n");
      exit(1);
    }

    if(aii > sum)
      bigger++;

  }

  if( !bigger )
  {
     printf("The matrix will not converge\n");
     exit(1);
  }
}


/******************************************************/
/* Read input from file */
void get_input(char filename[])
{
  FILE * fp;
  int i,j;

  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

  fscanf(fp,"%d ",&num);
  fscanf(fp,"%f ",&err);

  /* Now, time to allocate the matrices and vectors */
  a = (float**)malloc(num * sizeof(float*));
  if( !a)
  {
    printf("Cannot allocate a!\n");
    exit(1);
  }

  for(i = 0; i < num; i++)
  {
    a[i] = (float *)malloc(num * sizeof(float));
    if( !a[i])
    {
      printf("Cannot allocate a[%d]!\n",i);
      exit(1);
    }
  }

  x = (float *) malloc(num * sizeof(float));
  if( !x)
  {
    printf("Cannot allocate x!\n");
    exit(1);
  }


  b = (float *) malloc(num * sizeof(float));
  if( !b)
  {
    printf("Cannot allocate b!\n");
    exit(1);
  }

  /* Now .. Filling the blanks */

  /* The initial values of Xs */
  for(i = 0; i < num; i++)
    fscanf(fp,"%f ", &x[i]);

  for(i = 0; i < num; i++)
  {
    for(j = 0; j < num; j++)
      fscanf(fp,"%f ",&a[i][j]);

    /* reading the b element */
    fscanf(fp,"%f ",&b[i]);
  }

  fclose(fp);

}


// Calculates new value of x
float get_new_x(int rank)
{
  // value of the sum present in the formula
  float sum = 0;
  int j;
  for (j = 0; j < num; j++)
    if (rank != j)
    {
      if (rank == MASTER)
      {
        sum += a[MASTER][j]*x[j];
      } else
      {
        sum += a_row[j]*x[j];
      }
    }

  if (rank == MASTER)
  {
    return (b[rank]-sum)/a[MASTER][MASTER];
  } else
  {
    return (b[rank]-sum)/a_row[rank];
  }
}


/************************************************************/


int main(int argc, char *argv[])
{


  int i;
  int nit = 0; /* number of iterations */

  // Define variables needed for MPI

  int comm_sz, my_rank;

  // value of each new x computed
  float new_x;
  int error_flag;
  error_flag = 1;

  if( argc != 2)
  {
    printf("Usage: gsref filename\n");
    exit(1);
  }

  // Init MPI

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (IS_MASTER)
  {
    /* Read the input file and fill the global data structure above */
    get_input(argv[1]);

    // Check if the matrix will converge
    check_matrix();

    printf("Initial X guess");
    for (i = 0; i < num; i++)
      printf(" %f", x[i]);
    printf("\n");

    for (i = 1; i<comm_sz; i++)
    {
      MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(a[i], num, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
    }

    // Allocate x_old variable for calculating absolute error in master process

    x_old = (float *) malloc(num * sizeof(float));

  } else
  {

    // Receive num variable

    MPI_Recv(&num, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Since we did not call get_input for slave processes, we need to malloc
    // all the necessary pointers (a,b,x)

    allocate_matrices_slave();

    MPI_Recv(a_row, num, MPI_FLOAT, MASTER, 1, MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  }

  MPI_Bcast(b, num, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  printf("Received all input for %d\n", my_rank);

  while (error_flag)
  {
    MPI_Bcast(x, num, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    if (IS_MASTER)
    {
      memcpy(x_old, x, num * sizeof(float));
      nit++;
      x[0] = get_new_x(MASTER);
      for (i = 1; i < num; i++)
        MPI_Recv(&x[i], 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else
    {
      new_x = get_new_x(my_rank);
      MPI_Send(&new_x, 1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
    }

    if (IS_MASTER)
    {
      printf("%d)", nit);
      for (i = 0; i < num; i++)
        printf(" %f", x[i]);
      printf("\n");
      error_flag = check_error();
    }

    MPI_Bcast(&error_flag, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

  }

  /* Writing to the stdout */
  /* Keep that same format */
  if (IS_MASTER)
  {
    for( i = 0; i < num; i++)
      printf("%f\n", x[i]);

    printf("total number of iterations: %d\n", nit);
  }

  MPI_Finalize();

  exit(0);

}
