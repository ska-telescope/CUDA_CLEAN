
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
#include <stdbool.h>
#include <float.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "deconvolution.h"

// keeps track of running sum of model sources found
__device__ double d_flux = 0.0;
__device__ bool d_exit_early = false;
__device__ int d_source_counter = 0;

void init_config(Config *config)
{
	config->image_size = 1024;

	config->psf_size = 1024;

	config->number_minor_cycles = 60;

	config->loop_gain = 0.1; // 0.1 is typical

	config->gpu_max_threads_per_block = 1024;

	config->gpu_max_threads_per_block_dimension = 32;

	config->report_flux_each_cycle = false;

	config->weak_source_percent = 0.01; // example: 0.05 = 5%

	// Used to determine if we are extracting noise, based on the assumption
	// that located source < noise_detection_factor * running_average
	config->noise_detection_factor = 2.0;

	config->dirty_input_image = "../sample_data/dirty_image_1024.csv";

	config->model_output_file = "../sample_data/extracted_sources.csv";

	config->residual_output_image = "../sample_data/residual_image_1024.csv";

	config->psf_input_file = "../sample_data/point_spread_function_1024.csv";
}

void allocate_resources(PRECISION **dirty_image, Source **model, PRECISION **psf,
	unsigned int image_size, unsigned int psf_size, unsigned int num_minor_cycles)
{
	unsigned int image_size_squared = image_size * image_size;
	*dirty_image = (PRECISION*) calloc(image_size_squared, sizeof(PRECISION));
	*psf = (PRECISION*) calloc(psf_size * psf_size, sizeof(PRECISION));
	*model = (Source*) calloc(num_minor_cycles, sizeof(Source));
}

int performing_deconvolution(Config *config, PRECISION *dirty_image, Source *model, PRECISION *psf)
{
	PRECISION *d_residual;
	PRECISION3 *d_sources;
	PRECISION3 *d_max_locals;
	PRECISION *d_psf;

	// copying dirty image over to gpu and labeled now as the residual image
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
	bool exit_early = false;

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
			(d_sources, d_max_locals, cycle_number, config->image_size, config->loop_gain, flux, 
			 config->weak_source_percent, config->noise_detection_factor);
		cudaDeviceSynchronize();

		subtract_psf_from_residual<<<psf_blocks, psf_threads>>>
				(d_residual, d_sources, d_psf, cycle_number, config->image_size, config->psf_size, config->loop_gain);
		cudaDeviceSynchronize();

		if(config->report_flux_each_cycle)
		{
			cudaMemcpyFromSymbol(&flux, d_flux, sizeof(double), 0, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
		}

		compress_sources<<<1, 1>>>(d_sources);

		cudaMemcpyFromSymbol(&exit_early, d_exit_early, sizeof(bool), 0, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		if(exit_early)
		{
			printf(">>> UPDATE: Terminating minor cycles as now just cleaning noise...\n\n");
			break;
		}

		if(config->report_flux_each_cycle)
			printf(">>> INFO: Cycle %d reports flux: %.5e\n\n", cycle_number, flux);

		++cycle_number;
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf(">>> GPU Hogbom completed in %f ms for total cycles %d (average %f ms per cycle)...\n\n", 
		milliseconds, cycle_number, milliseconds / cycle_number);

	int number_of_sources_found = 0.0;
	cudaMemcpyFromSymbol(&number_of_sources_found, d_source_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	printf(">>> UPDATE: Copying %d extracted sources back from the GPU...\n\n", number_of_sources_found);
	CUDA_CHECK_RETURN(cudaMemcpy(model, d_sources, number_of_sources_found * sizeof(Source),
 		cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy(dirty_image, d_residual, sizeof(PRECISION) * image_size_square, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	return number_of_sources_found;
}

__global__ void find_max_source_row_reduction(const PRECISION *residual, PRECISION3 *local_max, const int image_size)
{
	unsigned int row_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(row_index >= image_size)
		return;

	// l, m, intensity 
	// just going to borrow the "m" or y coordinate and use to find the average in this row.
	//PRECISION3 max = MAKE_PRECISION3(0.0, (double) row_index, residual[row_index * image_size]);
	PRECISION3 max = MAKE_PRECISION3(0.0, ABS(residual[row_index * image_size]), residual[row_index * image_size]);
	PRECISION current;

	for(int col_index = 1; col_index < image_size; ++col_index)
	{
		current = residual[row_index * image_size + col_index];
		max.y += ABS(current);
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
	const int image_size, const PRECISION loop_gain, const double flux, const double weak_source_percent,
	const double noise_detection_factor)
{
	unsigned int col_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(col_index >= 1) // only single threaded
		return;

	//obtain max from row and col and clear the y (row) coordinate.
	PRECISION3 max = local_max[0];
	PRECISION running_avg = local_max[0].y;
	max.y = 0.0;

	PRECISION3 current;
	
	for(int index = 1; index < image_size; ++index)
	{
		current = local_max[index];
		running_avg += current.y;		
		current.y = index;

		if(ABS(current.z) > ABS(max.z))
			max = current;
	}

	running_avg /= (image_size * image_size);
	max.z *= loop_gain;
	
	// determine whether we drop out and ignore this source
	bool extracting_noise = max.z < noise_detection_factor * running_avg * loop_gain;
	bool weak_source = max.z < sources[0].z * weak_source_percent;
	d_exit_early = extracting_noise || weak_source;

	if(d_exit_early)
		return;

	// source was reasonable, so we keep it
	sources[d_source_counter] = max;
	d_flux = flux + max.z;
	++d_source_counter;
}

__global__ void compress_sources(PRECISION3 *sources)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index >= 1) // only single threaded
		return;

	PRECISION3 last_source = sources[d_source_counter - 1];
	for(int i = d_source_counter - 2; i >= 0; --i)
	{
		if((int)last_source.x == (int)sources[i].x && (int)last_source.y == (int)sources[i].y)
		{
			sources[i].z += last_source.z;
			--d_source_counter;
			break;
		}
	}
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
		sources[d_source_counter-1].x - half_psf_size + idx,
		sources[d_source_counter-1].y - half_psf_size + idy
	);
	
	// image coordinates fall out of bounds
	if(image_coord.x < 0 || image_coord.x >= image_size || image_coord.y < 0 || image_coord.y >= image_size)
		return;

	// Get required psf sample for subtraction
	const PRECISION psf_weight = psf[idy * psf_size + idx];

	// Subtract shifted psf sample from residual image
	residual[image_coord.y * image_size + image_coord.x] -= psf_weight  * sources[d_source_counter-1].z;
}


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
		}
	}

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

void save_sources_to_file(Source *source, int number_of_sources, char *output_file)
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

void load_sources_from_file(Source **source, int *number_of_sources, char *input_file)
{
	FILE *file = fopen(input_file, "r");

	if(file == NULL)
	{
		printf(">>> ERROR: Unable to load sources from file...\n\n");
		return;
	}

	// Determine number of sources to read in
	fscanf(file, "%d\n", number_of_sources);
	// Attempt to allocate memory
	*source = (Source*) calloc((*number_of_sources), sizeof(Source));
	// Determine if allocation failed
	if(*source == NULL)
	{
		fclose(file);
		return;
	}

	PRECISION temp_l = 0.0;
	PRECISION temp_m = 0.0;
	PRECISION temp_intensity = 0.0;

	for(int index = 0; index < (*number_of_sources); ++index)
	{
		#if SINGLE_PRECISION
			fscanf(file, "%f %f %f\n", &temp_l, &temp_m, &temp_intensity);
		#else
			fscanf(file, "%lf %lf %lf\n", &temp_l, &temp_m, &temp_intensity);
		#endif

		(*source)[index] = (Source) {
			.l = temp_l,
			.m = temp_m,
			.intensity = temp_intensity
		};
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

//**************************************//
//      UNIT TESTING FUNCTIONALITY      //
//**************************************//

void unit_test_init_config(Config *config)
{
	config->image_size = 1024;
	config->psf_size = 1024;
	config->number_minor_cycles = 60;
	config->loop_gain = 0.1;
	config->gpu_max_threads_per_block = 1024;
	config->gpu_max_threads_per_block_dimension = 32;
	config->report_flux_each_cycle = false;
	config->weak_source_percent = 0.0; // never exit early
	config->noise_detection_factor = 2.0;
	config->dirty_input_image = "../unit_test_data/dirty_image_1024.csv";
	config->psf_input_file = "../unit_test_data/point_spread_function_1024.csv";
}

PRECISION unit_test_extract_sources(Config *config)
{
	PRECISION error = (SINGLE_PRECISION) ? FLT_MAX : DBL_MAX;

	PRECISION *residual = NULL;
	Source *model = NULL;
	PRECISION *psf = NULL;
	allocate_resources(&residual, &model, &psf, config->image_size,
		config->psf_size, config->number_minor_cycles);

	if(residual == NULL || model == NULL || psf == NULL)
	{
		printf(">>> ERROR: Unable to allocate required resources, terminating...\n\n");
		clean_up(&residual, &model, &psf);
		return error;
	}

	bool loaded_dirty = load_image_from_file(residual, config->image_size, config->dirty_input_image);
	bool loaded_psf = load_image_from_file(psf, config->psf_size, config->psf_input_file);

	if(!loaded_dirty || !loaded_psf)
	{
		printf(">>> ERROR: Unable to load dirty or psf image from file, terminating...\n\n");
		clean_up(&residual, &model, &psf);
		return error;
	}

	int num_sources_found = performing_deconvolution(config, residual, model, psf);

	Source *reference_sources = NULL;
	int number_of_reference_sources = 0;
	load_sources_from_file(&reference_sources, &number_of_reference_sources, "../unit_test_data/reference_sources.csv");

	if(reference_sources == NULL)
	{
		printf(">>> ERROR: Unable to load reference sources from file, terminating...\n\n");
		clean_up(&residual, &model, &psf);
		return error;
	}

	if(number_of_reference_sources != num_sources_found)
	{
		printf(">>> ERROR: Number of reference sources (%d) does not match number of found \
			sources (%d), terminating...\n\n", number_of_reference_sources, num_sources_found);
		free(reference_sources);
		clean_up(&residual, &model, &psf);
		return error;
	}

	Source *extracted = (Source*) calloc(num_sources_found, sizeof(Source));

	if(extracted == NULL)
	{
		printf(">>> ERROR: Unable to reduce model source buffer for evaluation, terminating...\n\n");
		free(reference_sources);
		clean_up(&residual, &model, &psf);
		return error;
	}

	memcpy(extracted, model, sizeof(Source) * num_sources_found);

	qsort(extracted, num_sources_found, sizeof(Source), compare_sources);
	qsort(reference_sources, number_of_reference_sources, sizeof(Source), compare_sources);

	PRECISION max_difference = -DBL_MAX;

	for(int src_index = 0; src_index < num_sources_found; ++src_index)
	{
		PRECISION temp_diff = ABS(extracted[src_index].intensity - reference_sources[src_index].intensity);

		if(temp_diff > max_difference)
			max_difference = temp_diff;
	}

	free(reference_sources);
	free(extracted);
	clean_up(&residual, &model, &psf);

	return max_difference;	
}

int compare_sources(const void *a, const void *b)
{
	const Source *src_a = (Source*) a;
	const Source *src_b = (Source*) b;

	// dealing with conflict on l coordinate
	if(src_a->l == src_b->l)
		return (src_a->m > src_b->m) - (src_a->m < src_b->m);
	else
		return (src_a->l > src_b->l) - (src_a->l < src_b->l);
}