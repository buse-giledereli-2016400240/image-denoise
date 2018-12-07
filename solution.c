//mpicc -g solution.c -o solution
//mpiexec -n 5 ./solution argv[1]:input.txt argv[2]:output.txt argv[3]:0.6 argv[4]:0.1
//mpiexec -n 4 project.exe input.txt output.txt 0.6 0.1 


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char** argv) {
  //initialize the MPI environment
  MPI_Init(NULL, NULL);
  //find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  //find world_size from command line arguments 
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int N = world_size-1; //no of slave processes
  int i, j, k;
  double beta_value, pi_value, gamma_value;
  srand48(time(NULL)); //needed as a seed to rand48 function
  srand(time(NULL)); 
  
  int iter_count = 1000000; //iterations needed for denoising

  if (world_rank == 0) {
    printf("N: %d\n", N);
    
    int arr[200][200]; //opening array of the noised image

    FILE *f;
    f = fopen(argv[1],"r"); //open the input file
    if (f == NULL){
      printf("Error reading file!\n");
      exit (1);
    }
    for (i = 0; i < 40000; i++){
        fscanf(f, "%d", &arr[i/200][i%200]); //the text file is converted to a 2 dimentional array
    }
    fclose(f);

    beta_value = atof(argv[3]); //get the beta value from command line arguments
    pi_value = atof(argv[4]); //get the pi value from command line arguments
    gamma_value = 0.5*log((1-pi_value)/pi_value);

    for(i = 1 ; i <= N ; i++){
      MPI_Send(arr[200/N*(i-1)], 200/N*200, MPI_INT, i, 0, MPI_COMM_WORLD); //each slave processor receives some part of the array
      MPI_Send(&beta_value, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
      MPI_Send(&gamma_value, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }

    for(i = 1 ; i <= N ; i++){
      MPI_Recv(arr[200/N*(i-1)], 200/N*200, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //receives back the denoised array
    }

    FILE *w = fopen(argv[2], "w");
    if (w == NULL){
      printf("Error writing file!\n");
      exit(1);
    }

    for(i = 0; i < 200; i++){
      for(j = 0; j < 200; j++){
        fprintf(w, "%d ", arr[i][j]);
      }
      fprintf(w, "\n");
    }

    fclose(f);

  }
  else{
    int row = 200/N;
    int col = 200;
    int subarr[row][col]; //sub array of the processor
    MPI_Recv(subarr, row*col, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&beta_value, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&gamma_value, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int top[200]; //holds the row on top of the boundary
    int bottom[200]; //holds the row on bottom of the boundary

    //printf("Process %d received elements \n", world_rank);

    int z_arr[200/N][200];
    for(i = 0; i < 200/N; i++){
      for(j = 0; j < 200; j++){
        z_arr[i][j] = subarr[i][j];
      }
    }
    //this is the part that does the Metropolis-Hastings algorithm
    for(i = 0; i < iter_count/N; i++){

      //to avoid any deadlocks odd numbered processors send data first, even numbered processors receive data first
      if(world_rank % 2 == 0){
        //this is for the middle processors
        if(world_rank > 1 && world_rank < N){
          MPI_Send(z_arr[0], col, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD); //tag 0 means process is sending its first row
          MPI_Send(z_arr[row-1], col, MPI_INT, world_rank+1, 1, MPI_COMM_WORLD); //tag 1 means process is sending its last row
          MPI_Recv(top, col, MPI_INT, world_rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //tag 1 means process is receiving its top row
          MPI_Recv(bottom, col, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //tag 0 means process is receiving its bottom row
        }
        //this is for first processor
        else if(world_rank == 1){
          MPI_Send(z_arr[row-1], col, MPI_INT, world_rank+1, 1, MPI_COMM_WORLD); //tag 1 means process is sending its last row
          MPI_Recv(bottom, col, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //tag 0 means process is receiving its bottom row
        }
        //this is for last processor
        else{
          MPI_Send(z_arr[0], col, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD); //tag 0 means process is sending its first row
          MPI_Recv(top, col, MPI_INT, world_rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //tag 1 means process is receiving its top row
        }
        
      }
      else{
        //this is for the middle processors
        if(world_rank > 1 && world_rank < N){
          MPI_Recv(top, col, MPI_INT, world_rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //tag 1 means process is receiving its top row
          MPI_Recv(bottom, col, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //tag 0 means process is receiving its bottom row
          MPI_Send(z_arr[0], col, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD); //tag 0 means process is sending its first row
          MPI_Send(z_arr[row-1], col, MPI_INT, world_rank+1, 1, MPI_COMM_WORLD); //tag 1 means process is sending its last row
        }
        //this is for first processor
        else if(world_rank == 1){
          MPI_Recv(bottom, col, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //tag 0 means process is receiving its bottom row
          MPI_Send(z_arr[row-1], col, MPI_INT, world_rank+1, 1, MPI_COMM_WORLD); //tag 1 means process is sending its last row
        }
        //this is for last processor
        else{
          MPI_Recv(top, col, MPI_INT, world_rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //tag 1 means process is receiving its top row
          MPI_Send(z_arr[0], col, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD); //tag 0 means process is sending its first row
        }
      }
      

      int r = rand() % row; //random row
      int c = rand() % col; //random column
      int sum = 0;

      for(j = r-1; j < r+2; j++){
        for(k = c-1; k < c+2; k++){
          if(j >= 0 && j < row && k >= 0 && k < col){
            sum += z_arr[j][k]; //we sum all neighbours and the selected node
          }
          else if(j < 0 && world_rank > 1){
            sum += top[k];
          }
          else if(j >= row && world_rank < N){
            sum += bottom[k];
          }
        }
      }
      sum = sum - z_arr[r][c]; //we subtract the selected node since it is not a neighbour
      double delta_E = -2*gamma_value*subarr[r][c]*z_arr[r][c] - 2*beta_value*z_arr[r][c]*sum;
      if (log(drand48()) < delta_E){
        z_arr[r][c] = - z_arr[r][c];
      }
    }

    MPI_Send(z_arr, row*col, MPI_INT, 0, 0, MPI_COMM_WORLD);    

    //for(i = 0 ; i < 200/N ; i++)
    //  printf("%d ", subarr[i][0]);
    //printf("\n");
  }
  MPI_Finalize();
}