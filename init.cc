/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2021                                                     */
/*********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef U18
#include <cblas.h>
#else
#include <cblas-openblas64.h>
#endif

#include <omp.h>

#include "main.h"
#include "init.h"


/*-------------------------------------------------------------------------------*/
/* Initialisation of local matrixes A, B and C                                   */
/* Each process initializes its local parts of matrixes: simulates a parallel    */
/* initialization from files on disks.                                           */
/*-------------------------------------------------------------------------------*/
void LocalMatrixInit(void)
{
 int i, j;                                /* Local matrix indexes                */

/* Initialization of the local matrix elements                                   */
 for (i = 0; i < SIZE; i++)
    for (j = 0; j < SIZE; j++)
       A[i][j] = (T_real) (0.00001*i*SIZE + 0.000002*j);

 for (i = 0; i < SIZE; i++)
    for (j = 0; j < SIZE; j++) {
       B[i][j]  = (T_real) (0.0001*i*SIZE + 0.0000003*j);
       TB[j][i] = (T_real) (0.0001*i*SIZE + 0.0000003*j);
    }

 for (i = 0; i < SIZE; i++)
    for (j = 0; j < SIZE; j++)
       C[i][j] = 0.0;
}


/*-------------------------------------------------------------------------------*/
/* Command Line parsing.                                                         */
/*-------------------------------------------------------------------------------*/
void usage(int ExitCode, FILE *std)
{
 fprintf(std,"MatrixProduct usage: \n");
 fprintf(std,"\t [-h]: print this help\n");
 fprintf(std,"\t [-t <GPU(default)|CPU>]: run computations on target GPU or on target CPU\n");
 fprintf(std,"\t [-cpu-k <CPU kernel Id [0(default) - %d]>]\n",(NB_OF_CPU_KERNELS-1));
 fprintf(std,"\t [-cpu-nt <number of OpenMP threads> (default %d)]\n",DEFAULT_NB_THREADS);
 fprintf(std,"\t [-gpu-k <GPU kernel Id [0(default) - %d]>]\n",(NB_OF_GPU_KERNELS-1));
 fprintf(std,"\t [-no-check]: stops the results from being checked (suggested for performance measurements)\n");

 exit(ExitCode);
}


void CommandLineParsing(int argc, char *argv[])
{
 // Default init
 NbThreads = DEFAULT_NB_THREADS;
 OnGPUFlag = DEFAULT_ONGPUFLAG;
 CPUKernelId = DEFAULT_CPUKID;
 GPUKernelId = DEFAULT_GPUKID;

 // Init from the command line
 argc--; argv++;
 while (argc > 0) {
     if (strcmp(argv[0],"-t") == 0) {
       argc--; argv++;
       if (argc > 0) {
         if (strcmp(argv[0],"GPU") == 0) {
           OnGPUFlag = 1;
           argc--; argv++;
         } else if (strcmp(argv[0],"CPU") == 0) {
           OnGPUFlag = 0;
           argc--; argv++;
         } else {
           fprintf(stderr,"Error: unknown computation target '%s'!\n",argv[0]);
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }

     } else if (strcmp(argv[0],"-cpu-k") == 0) {
       argc--; argv++;
       if (argc > 0) {
         CPUKernelId = (ckid_t) atoi(argv[0]);
         argc--; argv++;
         if (CPUKernelId < 0 || CPUKernelId >= NB_OF_CPU_KERNELS) {
           fprintf(stderr,"Error: CPU kernel Id has to in [0 - %d]!\n",(NB_OF_CPU_KERNELS-1));
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }

     } else if (strcmp(argv[0],"-cpu-nt") == 0) {
       argc--; argv++;
       if (argc > 0) {
         NbThreads = atoi(argv[0]);
         argc--; argv++;
         if (NbThreads <= 0) {
           fprintf(stderr,"Error: number of thread has to be >= 1!\n");
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }

     } else if (strcmp(argv[0],"-gpu-k") == 0) {
       argc--; argv++;
       if (argc > 0) {
         GPUKernelId = (gkid_t) atoi(argv[0]);
         argc--; argv++;
         if (GPUKernelId < 0 || GPUKernelId >= NB_OF_GPU_KERNELS) {
           fprintf(stderr,"Error: GPU kernel Id has to in [0 - %d]!\n",(NB_OF_GPU_KERNELS-1));
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }

     } else if (strcmp(argv[0],"-no-check") == 0) {
       argc--; argv++;
       check_results = 0;

     } else if (strcmp(argv[0],"-h") == 0) {
       usage(EXIT_SUCCESS, stdout);
     } else {
       usage(EXIT_FAILURE, stderr);
     }
 }

 // Complementary inits
 openblas_set_num_threads(1);                    // Set OpenBLAS in sequential mode
}


/*-------------------------------------------------------------------------------*/
/* Print result of the parallel computation and performances                     */
/*-------------------------------------------------------------------------------*/
void PrintResultsAndPerf(double dk, double dt, double dkt,
                         double gfk, double gfkt, double bwt, int ongpu)
{
 //fprintf(stdout,"- Results:\n");
 fprintf(stdout,"\n- Examples of results:\n\t C[%d][%d] = %f\n",
         0,SIZE-1,(float) C[0][SIZE-1]);
 fprintf(stdout,"\t C[%d][%d] = %f\n",
         SIZE/2,SIZE/2,(float) C[SIZE/2][SIZE/2]);
 fprintf(stdout,"\t C[%d][%d] = %f\n",
         SIZE-1,0,(float) C[SIZE-1][0]);

 fprintf(stdout,"\n- Performance:\n");
 if(ongpu) {
     fprintf(stdout,"\t Complete Matrix Product:\n");
     fprintf(stdout,"\t   - Elapsed time = %f (s)\n", (float) dkt);
     fprintf(stdout,"\t   - Gflops = %f \n", (float) gfkt);
     fprintf(stdout,"\t Kernel computation:\n");
     fprintf(stdout,"\t   - Elapsed time = %f (s)\n", (float) dk);
     fprintf(stdout,"\t   - Gflops = %f \n", (float) gfk);
     fprintf(stdout,"\t Data transfers:\n");
     fprintf(stdout,"\t   - Elapsed time = %f (s)\n", (float) dt);
     fprintf(stdout,"\t   - BW           = %f (GB/s)\n", (float) bwt);
 } else {
     fprintf(stdout,"\t Complete Matrix Product:\n");
     fprintf(stdout,"\t   - Elapsed time = %f (s)\n", (float) dkt);
     fprintf(stdout,"\t   - Gflops = %f \n", (float) gfkt);
 }
 
 fflush(stdout);

}

/*-------------------------------------------------------------------------------*/
/* Result checking                                                               */
/*-------------------------------------------------------------------------------*/
T_real C_check[SIZE][SIZE];
// Different values for epsilon depending if we use float or double
#ifdef DP
#define EPSILON 1e-14
#else
#define EPSILON 1e-4
#endif

void CheckResults(void) {

    fprintf(stdout,"\n- Checking results (comparison with CPU BLAS):\n");

    // Recomputing the matrix product on CPU
    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel
   {
     int reste = SIZE % omp_get_num_threads();
     int quotient = SIZE / omp_get_num_threads();
     int NbLig = quotient +
                 (omp_get_thread_num() < reste ? 1 : 0);
     int offsetLig = quotient*omp_get_thread_num() +
                     (omp_get_thread_num() < reste ? omp_get_thread_num() : reste);
     CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NbLig, SIZE, SIZE,
                1.0, &A[offsetLig][0], SIZE,
                &B[0][0], SIZE,
                0.0, &C_check[offsetLig][0], SIZE);
   }

   // Comparing the different results
   // - maximum difference
   double max_diff = 0.0;
   // - position with the largest relative difference
   int max_X = 0;
   int max_Y = 0;
   // - epsilon for relative differences
   double epsilon = EPSILON;
   // - number of cases where the error is too large
   int cases = 0;

   for(int i = 0; i < SIZE; ++i){
       for(int j = 0; j < SIZE; ++j){
           double diff = fabs(C[i][j] - C_check[i][j]); //difference between results
           double standard = fabs(C_check[i][j]);

           // Checks if the difference is large relative to the expected result
           if (diff > standard*epsilon)
               ++cases; // Register the case
           if (standard > 0.0 && diff/standard > max_diff){ // Store the largest difference seen so far
                   max_diff = diff/standard;
                   max_X = i;
                   max_Y = j;
           }
       }
   }

   if(cases == 0){
       fprintf(stdout,"The results are correct for %s with a precision of %.5e.\n", T_REAL_TEXT, epsilon);
       fprintf(stdout,"Maximum relative difference encountered: %.5e.\n", max_diff);
   } else {
       fprintf(stdout,"*** WARNING ***\n");
       fprintf(stdout,"The results are incorrect for %s with a precision of %.5e.\n", T_REAL_TEXT, epsilon);
       fprintf(stdout,"Number of cell with imprecise results: %d\n", cases);
       fprintf(stdout,"Cell C[%d][%d] contained the largest relative difference of %.5e\n", max_X, max_Y, max_diff);
       fprintf(stdout,"Expected value: %15.15lf\n", C_check[max_X][max_Y]);
       fprintf(stdout,"Computed value: %15.15lf\n", C[max_X][max_Y]);
   }
}

