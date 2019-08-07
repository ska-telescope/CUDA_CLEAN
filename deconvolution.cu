
// Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.  

#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "deconvolution.h"

// keeps track of running sum of model sources found
__device__ double d_flux = 0.0;

void init_config(Config *config)
{
	config->image_size = 1024;

	config->psf_size = 1024;

	config->number_minor_cycles = 60;

	config->loop_gain = 0.1; // 0.1 is typical

	config->gpu_max_threads_per_block = 1024;

	config->gpu_max_threads_per_block_dimension = 32;

	config->dirty_input_image = "../images/sample_dirty_image_1024_norm.csv";

	config->model_output_file = "../images/sample_output_sources.csv";

	config->residual_output_image = "../images/sample_output_residual.csv";

	config->psf_input_file = "../images/sample_psf_1024_norm.csv";
}

void allocate_resources(PRECISION **dirty_image, Source **model, PRECISION **psf,
	unsigned int image_size, unsigned int psf_size, unsigned int num_minor_cycles)
{
	unsigned int image_size_squared = image_size * image_size;
	*dirty_image = (PRECISION*) calloc(image_size_squared, sizeof(PRECISION));
	*psf = (PRECISION*) calloc(psf_size * psf_size, sizeof(PRECISION));
	*model = (Source*) calloc(num_minor_cycles, sizeof(Source));
}

void performing_deconvolution(Config *config, PRECISION *dirty_image, Source *model, PRECISION *psf)
{
	PRECISION *d_residual;
	PRECISION3 *d_sources;
	PRECISION3 *d_max_locals;
	PRECISION *d_psf;

	//copying dirty image over to gpu and labeled now as the residual image
	int image_size_square = config->image_size * config->image_size;
	CUDA_CHECK_RETURN(cudaMalloc(&d_residual, sizeof(PRECISION) * image_size_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_residual, dirty_image, sizeof(PRECISION) * image_size_square, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	//copy the psf over to GPU
	int psf_size_square = config->psf_size * config->psf_size;
	CUDA_CHECK_RETURN(cudaMalloc(&d_psf, sizeof(PRECISION) * psf_size_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_psf, psf, sizeof(PRECISION) * psf_size_square, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	//copying source list over to GPU - size of the number of minor cycles
	CUDA_CHECK_RETURN(cudaMalloc(&d_sources, sizeof(PRECISION3) * config->number_minor_cycles));
	CUDA_CHECK_RETURN(cudaMalloc(&d_max_locals, sizeof(PRECISION3) * config->image_size));	

	// row reduction configuration
	int max_threads_per_block = min(config->gpu_max_threads_per_block, config->image_size);
	int num_blocks = (int) ceil((double) config->image_size / max_threads_per_block);
	dim3 reduction_blocks(num_blocks, 1, 1);
	dim3 reduction_threads(config->gpu_max_threads_per_block, 1, 1);

	// PSF subtraction configuration
	int max_psf_threads_per_block_dim = min(config->gpu_max_threads_per_block_dimension, config->psf_size);
	int num_blocks_psf = (int) ceil((double) config->psf_size / max_psf_threads_per_block_dim);
	dim3 psf_blocks(num_blocks_psf, num_blocks_psf, 1);
	dim3 psf_threads(max_psf_threads_per_block_dim, max_psf_threads_per_block_dim, 1);

	int cycle_number = 0;
	double flux = 0.0;
	//starting timer here
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	while(cycle_number < config->number_minor_cycles)
	{
		// Find local row maximum via reduction
		find_max_source_row_reduction<<<reduction_blocks, reduction_threads>>>
			(d_residual, d_max_locals, config->image_size);
		cudaDeviceSynchronize();

		// Find final image maximum via column reduction (local maximums array)
		find_max_source_col_reduction<<<1, 1>>>
			(d_sources, d_max_locals, cycle_number, config->image_size, config->loop_gain, flux);
		cudaDeviceSynchronize();

		subtract_psf_from_residual<<<psf_blocks, psf_threads>>>
				(d_residual, d_sources, d_psf, cycle_number, config->image_size, config->psf_size, config->loop_gain);
		cudaDeviceSynchronize();

  		// cudaMemcpyFromSymbol(&flux, d_flux, sizeof(double), 0, cudaMemcpyDeviceToHost);
		// cudaDeviceSynchronize();

		//printf(">>> INFO: Cycle %d reports flux: %.5e\n\n", cycle_number, flux);

		++cycle_number;
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf(">>> GPU Hogbom completed in %f ms for total cycles %d (average %f ms per cycle)...\n\n", 
		milliseconds, cycle_number, milliseconds / cycle_number);

	printf(">>> UPDATE: Copying extracted sources back from the GPU...\n\n");
	CUDA_CHECK_RETURN(cudaMemcpy(model, d_sources, config->number_minor_cycles * sizeof(Source),
 		cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	// for(int i = 0 ; i < config->number_minor_cycles; ++i)
	// {
	// 	printf("Source %d found at (l, m) %f, %f with value (intensity) %.15f...\n\n",
	// 		i, model[i].l, model[i].m, model[i].intensity);
	// }


	CUDA_CHECK_RETURN(cudaMemcpy(dirty_image, d_residual, sizeof(PRECISION) * image_size_square, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}

__global__ void find_max_source_row_reduction(const PRECISION *residual, PRECISION3 *local_max, const int image_size)
{
	unsigned int row_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(row_index >= image_size)
		return;

	// l, m, intensity
	PRECISION3 max = MAKE_PRECISION3(0.0, (double) row_index, residual[row_index * image_size]);
	PRECISION current;

	for(int col_index = 1; col_index < image_size; ++col_index)
	{
		current = residual[row_index * image_size + col_index];
		if(ABS(current) > ABS(max.z))
		{
			// update m and intensity
			max.x = (double) col_index;
			max.z = current;
		}
	}

	local_max[row_index] = max;
}

__global__ void find_max_source_col_reduction(PRECISION3 *sources, const PRECISION3 *local_max, const int cycle_number,
	const int image_size, const PRECISION loop_gain, const double flux)
{
	unsigned int col_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(col_index >= 1) // only single threaded
		return;

	PRECISION3 max = local_max[0];
	PRECISION3 current;

	for(int index = 1; index < image_size; ++index)
	{
		current = local_max[index];
		if(ABS(current.z) > ABS(max.z))
			max = current;
	}

	sources[cycle_number] = max;
	sources[cycle_number].z *= loop_gain;
	d_flux = flux + sources[cycle_number].z;
}

__global__ void subtract_psf_from_residual(PRECISION *residual, PRECISION3 *sources, const PRECISION *psf, 
	const int cycle_number, const int image_size, const int psf_size, const PRECISION loop_gain)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	// thread out of bounds
	if(idx >= psf_size || idy >= psf_size)
		return;

	const int half_psf_size = psf_size / 2;

	// Determine image coordinates relative to source location
	int2 image_coord = make_int2(
		sources[cycle_number].x - half_psf_size + idx,
		sources[cycle_number].y - half_psf_size + idy
	);
	
	// image coordinates fall out of bounds
	if(image_coord.x < 0 || image_coord.x >= image_size || image_coord.y < 0 || image_coord.y >= image_size)
		return;

	// Get required psf sample for subtraction
	const PRECISION psf_weight = psf[idy * psf_size + idx];

	// Subtract shifted psf sample from residual image
	residual[image_coord.y * image_size + image_coord.x] -= psf_weight  * sources[cycle_number].z;


}

// __global__ void find_max_in_residual_image(PRECISION *residual, PRECISION3 *sources, int cycle_number, int image_size)
// {
// 	int idx = blockIdx.x*blockDim.x + threadIdx.x;
// 	int idy = blockIdx.y*blockDim.y + threadIdx.y;

// 	if(idx >= image_size || idy >= image_size)
// 		return;


// 	PRECISION val = residual[idy * image_size + idx];


// 	// NO ATOMIC MAX FOR ANY FLOATING POINT NUMBERS IN CUDA (our own?)
// 	int old_max = atomic_max(&(sources[cycle_number].z), val);
// 	if(old_max < val)
// 	{
// 		// NO ATOMIC EXCH FOR DOUBLES IN CUDA (make our own?)
// 		// https://stackoverflow.com/questions/12626096/why-has-atomicadd-not-been-implemented-for-doubles
// 		// https://www.cse-lab.ethz.ch/wp-content/uploads/2018/05/S3101-Atomic-Memory-Operations.pdf
// 		atomic_exchange(&(sources[cycle_number].x), (PRECISION) idx-image_size);
// 		atomic_exchange(&(sources[cycle_number].y), (PRECISION) idy-image_size);
// 	}

// }

// https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
// __device__ double atomic_max(double *address, double value)
// {
//     unsigned long long int ret = __double_as_longlong(*address);
//     while(value > __longlong_as_double(ret))
//     {
//         unsigned long long int old = ret;
//         if((ret = atomicCAS((unsigned long long int*)address, old, __double_as_longlong(value))) == old)
//             break;
//     }
//     return __longlong_as_double(ret);
// }

// __device__ double atomic_exchange(double *address, double value)
// {
// 	// get address as large int
// 	unsigned long long int* address_as_ull = (unsigned long long int*) address;
// 	// get value from large int address
// 	unsigned long long int old = *address_as_ull;
// 	unsigned long long int assumed;

// 	do {
// 		assumed = old;
// 		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
// 	} while (assumed != old);

// 	return __longlong_as_double(old);
// }

bool load_image_from_file(PRECISION *image, unsigned int size, char *input_file)
{
	FILE *file = fopen(input_file, "r");

	if(file == NULL)
	{
		printf(">>> ERROR: Unable to load image from file...\n\n");
		return false;
	}

	for(int row = 0; row < size; ++row)
	{
		for(int col = 0; col < size; ++col)
		{
			int image_index = row * size + col;

			#if SINGLE_PRECISION
				fscanf(file, "%f ", &(image[image_index]));
			#else
				fscanf(file, "%lf ", &(image[image_index]));
			#endif

			// printf("%f ", image[image_index]);
		}
		// printf("\n");
	}

	// printf("\n\n");

	fclose(file);
	return true;
}


void save_image_to_file(PRECISION *image, unsigned int size, char *real_file)
{
	FILE *image_file = fopen(real_file, "w");

	if(image_file == NULL)
	{
		printf(">>> ERROR: Unable to save image to file, moving on...\n\n");
		return;
	}

	for(int row = 0; row < size; ++row)
	{
		for(int col = 0; col < size; ++col)
		{
			unsigned int image_index = row * size + col;

			#if SINGLE_PRECISION
				fprintf(image_file, "%f ", image[image_index]);
			#else
				fprintf(image_file, "%.15f ", image[image_index]);
			#endif
		}

		fprintf(image_file, "\n");
	}

	fclose(image_file);
}

void save_sources_to_file(Source *source, unsigned int number_of_sources, char *output_file)
{
	FILE *file = fopen(output_file, "w");

	if(file == NULL)
	{
		printf(">>> ERROR: Unable to save sources to file, moving on...\n\n");
		return;
	}

	fprintf(file, "%d\n", number_of_sources);
	for(int index = 0; index < number_of_sources; ++index)
	{
		#if SINGLE_PRECISION
			fprintf(file, "%f %f %f\n", source[index].l, source[index].m, source[index].intensity);
		#else
			fprintf(file, "%.15f %.15f %.15f\n", source[index].l, source[index].m, source[index].intensity);
		#endif
	}

	fclose(file);
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;

	printf(">>> CUDA ERROR: %s returned %s at %s : %u ",statement, file, cudaGetErrorString(err), line);
	exit(EXIT_FAILURE);
}

void clean_up(PRECISION **dirty, Source **model, PRECISION **psf)
{
	if(*dirty) free(*dirty);
	if(*model) free(*model);
	if(*psf)   free(*psf);
}