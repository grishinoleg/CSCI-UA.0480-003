/* Oleg Grishin, NYU, 2014
  Parallel Computing, CSCI-UA.0480-003
  Remark: works for any p (including p's that aren't power of 2) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// Defines for more readable code

#define MASTER 0
#define IS_MASTER my_rank == MASTER

/***** Globals ******/
float *a; /* The coefficients */
float *x;  /* The unknowns */
float *x_old; /* Old value of unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */


/****** Function declarations */
void allocate_matrices_slave(int comm_sz, int row_num); /* Allocate matrices for
slave processes */
int check_error(); /* Check absolute relative error */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */
float get_new_x(float *row, int entry_num, int my_rank); /* pass pointer to what row to read
from, pass what entry it is (for i!=j comparison) and get x back */

/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/* Since get_input is not called by non-master processes, we allocate
a, b and x */

void allocate_matrices_slave(comm_sz, row_num)
{
  /* Keeping double pointer for simplicity - no need to rewrite calculations
  for master process in get_new_x() function */

  // Allocate several rows if num > comm_sz; just 1 otherwise

  if (num > comm_sz)
  {
    a = (float *)malloc(num * row_num * sizeof(float));
  }
  else
  {
    a = (float *)malloc(num * sizeof(float));
  }
  if( !a)
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
    aii = fabs(a[i*num + i]);

    for(j = 0; j < num; j++)
       if( j != i)
   sum += fabs(a[i*num + j]);

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
  a = (float*)malloc(num * num * sizeof(float*));
  if( !a)
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

  /* Now .. Filling the blanks */

  /* The initial values of Xs */
  for(i = 0; i < num; i++)
    fscanf(fp,"%f ", &x[i]);

  for(i = 0; i < num; i++)
  {
    for(j = 0; j < num; j++)
      fscanf(fp,"%f ",&a[i*num + j]);

    /* reading the b element */
    fscanf(fp,"%f ",&b[i]);
  }

  fclose(fp);

}


// Calculates new value of x
float get_new_x(float *row, int entry, int my_rank)
{
  // value of the sum present in the formula
  float sum = 0;
  int j;
  for (j = 0; j < num; j++)
    if (entry != j)
    {
      if (IS_MASTER)
      {
        // Since we rewrite x we need to use old values on master process
        sum += row[j]*x_old[j];
      } else
      {
        sum += row[j]*x[j];
      }
    }

  return (b[entry]-sum)/row[entry];

}


/************************************************************/


int main(int argc, char *argv[])
{


  int i, nit = 0; /* number of iterations */

  // Define variables needed for MPI and out computations

  int comm_sz, my_rank, error_flag = 1, rows_num = 1, my_entry_offset = 0;
  // rows_num is for storing how many rows each process gets
  // my_entry_offset is for calculating X and checking j!=i

  // Value of each new x computed

  float new_x;

  // Necessary group manipulations (i.e. more processes than groups)

  MPI_Group world_group;
  MPI_Group my_group;
  MPI_Comm my_comm;

  if( argc != 2)
  {
    printf("Usage: gsref filename\n");
    exit(1);
  }

  // Init MPI

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // Create a group from world comm

  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  // Initial input and num passing

  if (IS_MASTER)
  {
    /* Read the input file and fill the global data structure above */
    get_input(argv[1]);

    // Check if the matrix will converge
    check_matrix();

    // Send num variable to all the processes
    for (i = 1; i<comm_sz; i++)
      MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

  } else
  {
    // Receive num variable

    MPI_Recv(&num, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // If number of processes is larger than num of rows, make smaller comm

  if (num > comm_sz)
  {
    /* First processes get more processes, i.e. if num = 10 and comm_sz = 4,
    p0: 3 rows, p1: 3 rows, p2: 2 rows, p3: 2 rows */
    rows_num = num / comm_sz;
    if (my_rank < num % comm_sz)
      rows_num++;
    MPI_Comm_create (MPI_COMM_WORLD, world_group, &my_comm);
  } else
  {
    int needed_p_vals[num];
    for (i = 0; i < num; i++)
      needed_p_vals[i] = i;
    MPI_Group_incl (world_group, num, needed_p_vals, &my_group);
    MPI_Comm_create (MPI_COMM_WORLD, my_group, &my_comm);

    if (my_rank >= num)
    {
      // Quit processes that are not necessery
      MPI_Finalize();
      exit(0);
    }

    MPI_Comm_size(my_comm, &comm_sz);

  }

  // Used to map entry number back to processes
  int process_offsets[comm_sz];

  if (IS_MASTER)
  {
    int new_rows_num = num / comm_sz;
    if (num % comm_sz > 0)
    {
      new_rows_num++;
    }
    int new_rows_num_each = 0;
    process_offsets[0] = new_rows_num;

    /* If num > comm_sz, then we need to distribute several rows for each
    process; else we need to distribute rows to num processes */

    if (num > comm_sz)
    {

      for (i = 1; i<comm_sz; i++)
      {

        new_rows_num_each = num / comm_sz;

        if (i < num % comm_sz)
        {
          new_rows_num_each++;
        }

        // We need to recompute rows_num for global offset

        new_rows_num += new_rows_num_each;

        process_offsets[i] = new_rows_num;

        MPI_Send(&a[process_offsets[i-1]*num], num*new_rows_num_each,
          MPI_FLOAT, i, 0, my_comm);
      }
    } else
    {
      for (i = 1; i<comm_sz; i++)
      {
        MPI_Send(&a[i*num], num, MPI_FLOAT, i, 0, my_comm);
      }
    }


    // Allocate x_old variable for calculating absolute error in master process

    x_old = (float *) malloc(num * sizeof(float));

  } else
  {

    // Since we did not call get_input for slave processes, we need to malloc
    // all the necessary pointers (a,b,x)

    allocate_matrices_slave(comm_sz, rows_num);

    if (num > comm_sz)
    {
      // Receive several X's

      MPI_Recv(a, num * rows_num, MPI_FLOAT, MASTER, 0, my_comm,
        MPI_STATUS_IGNORE);

    } else
    {
      // Receive one X

      MPI_Recv(a, num, MPI_FLOAT, MASTER, 0, my_comm, MPI_STATUS_IGNORE);
    }

    // Calculate my_entry offset for i!=j computations

    if (num > comm_sz)
    {
      for (i = 0; i < my_rank; i++)
      {
        my_entry_offset += num / comm_sz;
        if (i < num % comm_sz)
        {
          my_entry_offset++;
        }
      }
    }

  }

  // Send value of b to all the processes

  MPI_Bcast(b, num, MPI_FLOAT, MASTER, my_comm);

  // Error flag is one if one of the x's is not within the limits

  while (error_flag)
  {
    MPI_Bcast(x, num, MPI_FLOAT, MASTER, my_comm);

    if (IS_MASTER)
    {
      // Copy new x into x_old for easier calculations for error_flag
      memcpy(x_old, x, num * sizeof(float));
      nit++;
    }

    if (num > comm_sz)
    {
      // Calculate X for each row that is passed to the process
      for (i = 0; i < rows_num; i++)
      {
        if (IS_MASTER)
        {
          // Compute X's that main processor is assigned
          x[i] = get_new_x(&a[i*num], i, MASTER);
        } else
        {
          // Send all new X's back to MASTER. Use tag for ordering
          new_x = get_new_x(&a[i*num], my_entry_offset+i, my_rank);
          MPI_Send(&new_x, 1, MPI_FLOAT, MASTER, my_entry_offset+i,
            my_comm);
        }
      }
    } else
    {
      if (IS_MASTER)
      {
        // Compute x on MASTER
        x[MASTER] = get_new_x(&a[MASTER], MASTER, MASTER);
      } else
      {
        // Compute x on slaves and send to master
        new_x = get_new_x(a, my_rank, my_rank);
        MPI_Send(&new_x, 1, MPI_FLOAT, MASTER, 0, my_comm);
      }
    }

    if (IS_MASTER)
    {
      if (num > comm_sz)
      {
        int process = 1;
        for (i = rows_num; i < num; i++)
        {
          int j;
          for (j = 0; j < comm_sz; j++)
          {
            if (i >= process_offsets[j])
            {
              process = j+1;
            }
          }
          MPI_Recv(&x[i], 1, MPI_FLOAT, process, i, my_comm, MPI_STATUS_IGNORE);
        }
      } else
      {
        for (i = 1; i < comm_sz; i++)
          MPI_Recv(&x[i], 1, MPI_FLOAT, i, 0, my_comm, MPI_STATUS_IGNORE);
      }

    }

    if (IS_MASTER)
      error_flag = check_error();

    // Broadcast error flag back to slaves
    MPI_Bcast(&error_flag, 1, MPI_INT, MASTER, my_comm);

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
