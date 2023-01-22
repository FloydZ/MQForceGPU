#include "check_sol.h"
#include "check_thread.h"
#include "guess.h"
#include "partial_eval.h"
#include "read_sys.h"

#include "cuda_util.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

//
#define NUM_THREADS (1 << (N - K))
#define BLOCK_DIM (NUM_THREADS > 128 ? 128 : NUM_THREADS)
#define GRID_DIM (NUM_THREADS / BLOCK_DIM)

//
#define KERNEL_SOLUTIONS (1u << 2u)
#define KERNEL_SHARED_SOLUTIONS (1u << 2u)


#define PRINT_SOL(X) printf("%lX\n", X)
// #define PRINT_SOL(X)

#define LOG(level, f_, ...) fprintf(stdout, (f_), ##__VA_ARGS__)
// #define LOG(level, f_, ...)

extern "C" double get_ms_time(void) {
	struct timeval timev;

	gettimeofday(&timev, NULL);
	return (double) timev.tv_sec * 1000 + (double) timev.tv_usec / 1000;
}

__device__ __constant__ uint32_t deg2_block[MAX_K * (MAX_K - 1) / 2];

//template<const uint32_t tpb>
//__global__ void guess(const uint32_t *deg1, uint32_t *result,
//		const uint32_t num_threads, const uint32_t k);

#include "kernel_opt_shared.cuh"


__device__
uint32_t eval(const uint32_t *sys, const uint64_t sol, const uint32_t N, const uint32_t M) {
  uint32_t i, j, pos = 0;
  uint32_t x[64], check = 0;

  for (uint32_t b = 0; b < M; b += 32) {
    uint32_t mask = (M - b) >= 32 ? 0xffffffff : ((1 << (M - b)) - 1);

    for (i = 0; i < N; i++)
      x[i] = ((sol >> i) & 1) ? mask : 0;

    // computing quadratic part
    for (j = 1; j < N; j++)
      for (i = 0; i < j; i++)
        check ^= sys[pos++] & x[i] & x[j];

    // computing linear part
    for (i = 0; i < N; i++)
      check ^= sys[pos++] & x[i];

    // constant part
    check ^= sys[pos++];
  }

  return check;
}

__global__
void check(uint64_t *sol_out, const uint32_t *sys, const uint32_t *result,
		   const uint32_t N, const uint32_t M, const uint32_t K) {
	const uint32_t tid = (blockIdx.x*blockDim.x + threadIdx.x);

	for (uint32_t i = 0; i < KERNEL_SOLUTIONS; i++) {
		const uint64_t x = uint64_t(tid << K) | result[tid*KERNEL_SOLUTIONS + i];
		if (eval(sys, x, N, M) == 0) {
			// printf("found %d %lu\n", tid, x);
			*sol_out = x;
			//atomicExch(sol_out, x);
			break;
		}
	}
}

static int cuda_device = 0;
static bool init = false;

extern "C" void setDevice(int device) {
	cuda_device = device;
	init = false;
}

extern "C"
uint64_t searchSolution(uint32_t *coefficients, 
						unsigned int number_of_variables,
						unsigned int number_of_equations) {

	if (!init) {
		double initTime = 0;
		initTime -= get_ms_time();

		// set to designated device
		// int test;
		CUDA_ASSERT(cudaSetDevice(cuda_device));
		// cudaGetDevice(&test);
		// assert(atoi(argv[1]) == test);

		initTime += get_ms_time();
		//LOG(INFO, "init time = %f\n", initTime);

		init = true;
	}

	double preTime = 0, memTime = 0, recvTime = 0, checkTime = 0, ctTime = 0;
	float kernelTime = 0;
	uint32_t solCount = 0, ctCount = 0;

	uint64_t res = UINT64_MAX;

	// create events here
	cudaEvent_t start, stop;
	CUDA_ASSERT(cudaEventCreate(&start));
	CUDA_ASSERT(cudaEventCreate(&stop));
	CUDA_ASSERT(cudaDeviceSynchronize());

	uint32_t N = number_of_variables;
	uint32_t M = number_of_equations;

	uint32_t K = 32;

	if (K > MAX_K)
		K = MAX_K;

	if (N <= K)
		K = N - 1;
	
	const uint32_t sizeofsys = (1 + N + N*(N-1)) * (1+(M > 32)) * sizeof(uint32_t);
	uint32_t *sys = pack_sys_data(coefficients, N, M), *dsys;
	CUDA_ASSERT(cudaMalloc(&dsys, sizeofsys));
	CUDA_ASSERT(cudaMemcpy(dsys, sys, sizeofsys, cudaMemcpyHostToDevice));

	preTime -= get_ms_time(); // partial evaluation

	cudaData<uint32_t> deg1((K + 1) * NUM_THREADS);

	partial_eval(sys, deg1.host, N, K);

	preTime += get_ms_time();

	memTime -= get_ms_time(); // initializing GPU memory space

	// initialize constant memory space for the quadratic part
	CUDA_ASSERT(cudaMemcpyToSymbol(deg2_block, sys, sizeof(uint32_t) * K * (K - 1) / 2));
	CUDA_ASSERT(cudaDeviceSynchronize());

	// initialize global memory space for the linear parts
	deg1.write();

	// initialize global memory space for the results of each threads
	cudaData<uint32_t> result(NUM_THREADS * KERNEL_SOLUTIONS);

	memTime += get_ms_time();

	// launch kernel function and measure the elapsed time
	cudaEventRecord(start, 0);

	// guess<<<GRID_DIM, BLOCK_DIM>>>(deg1.dev, result.dev, NUM_THREADS, K);
	guess<<<GRID_DIM, BLOCK_DIM>>>((const uint32_t *)deg1.dev, result.dev, uint32_t(NUM_THREADS), K);
	CUDA_ASSERT(cudaDeviceSynchronize());

	CUDA_ASSERT(cudaEventRecord(stop, 0));
	CUDA_ASSERT(cudaEventSynchronize(stop));

	CUDA_ASSERT(cudaEventElapsedTime(&kernelTime, start, stop));
	CUDA_ASSERT(cudaDeviceSynchronize());

	recvTime -= get_ms_time(); // copy the results of each thread to host

#if 1
	result.read();
#endif

	recvTime += get_ms_time();

	checkTime -= get_ms_time(); // check if the results are available

	uint64_t ans;

#if 0
	uint64_t *dans;
	CUDA_ASSERT(cudaMalloc(&dans, 8));
	check<<<GRID_DIM, BLOCK_DIM>>>
		(dans, dsys, result.dev, N, M, K);
	CUDA_ASSERT(cudaDeviceSynchronize());

	CUDA_ASSERT(cudaMemcpy(&ans, dans, 8, cudaMemcpyDeviceToHost));
	// 	assert(check_sol(sys, ans, N, M) == 0);
#else

	for (uint64_t i = 0; i < NUM_THREADS; i++) {
		for (uint32_t j = 0; j < KERNEL_SOLUTIONS; j++) {
			ans = result.host[i*KERNEL_SOLUTIONS + j];
			if (check_sol(sys, (i << K) | ans, N, M) == 0) {
				solCount++;
				LOG(INFO, "thread %lX ---------> one solution %X\n", i, ans);
				PRINT_SOL((i << K) | ans);

				res = (i << K) | ans;

				goto end;
			}
		}
	}
end:
#endif
	checkTime += get_ms_time();

	float totalTime = preTime + memTime + kernelTime + recvTime + checkTime;

	// print the time for each step
	//LOG(INFO, "partial ");
	//LOG(INFO, "mem ");
	//LOG(INFO, "kernel ");
	//LOG(INFO, "recv ");
	//LOG(INFO, "check #sol ");
	//LOG(INFO, "(mult sol: t #ct)\n");
	LOG(INFO, "%.3f ", preTime);
	LOG(INFO, "%.3f ", memTime);
	LOG(INFO, "%.3f ", kernelTime);
	LOG(INFO, "%.3f ", recvTime);
	LOG(INFO, "%.3f ", checkTime);
	LOG(INFO, "%u ", solCount);
	LOG(INFO, "(%.3f  %u) ", ctTime, ctCount);

	LOG(INFO, "%.3f \n", totalTime);
	// release memory spaces
	free(sys);

	return res;
}
