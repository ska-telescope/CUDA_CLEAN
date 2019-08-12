
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

#include <cstdlib>
#include <cstdio>

#include "deconvolution.h"

int main(int argc, char **argv)
{
	// Initialise the config
	Config config;
	init_config(&config);

	// Allocate memory for images (residual/model), and PSF kernel
	printf(">>> UPDATE: Allocating resources for deconvolution...\n\n");
	PRECISION *residual = NULL;
	Source *model = NULL;
	PRECISION *psf = NULL;
	allocate_resources(&residual, &model, &psf, config.image_size,
		config.psf_size, config.number_minor_cycles);

	if(residual == NULL || model == NULL || psf == NULL)
	{
		printf(">>> ERROR: Unable to allocate required resources, terminating...\n\n");
		clean_up(&residual, &model, &psf);
		return EXIT_FAILURE;
	}

	printf(">>> UPDATE: Loading dirty image and PSF into memory...\n\n");
	// Load dirty image into memory as residual
	bool loaded_dirty = load_image_from_file(residual, config.image_size, config.dirty_input_image);
	// Load point spread function kernel into memory
	bool loaded_psf = load_image_from_file(psf, config.psf_size, config.psf_input_file);

	if(!loaded_dirty || !loaded_psf)
	{
		printf(">>> ERROR: Unable to load dirty or psf image from file, terminating...\n\n");
		clean_up(&residual, &model, &psf);
		return EXIT_FAILURE;
	}

	// Perform CLEAN
	printf(">>> UPDATE: Performing deconvolution...\n\n");
	int number_of_cycles_completed = performing_deconvolution(&config, residual, model, psf);
	printf(">>> UPDATE: Deconvolution complete...\n\n");
	
	printf(">>> UPDATE: Saving model sources to file...\n\n");
	// Save model sources to file
	save_sources_to_file(model, number_of_cycles_completed, config.model_output_file);

	printf(">>> UPDATE: Saving residual output image to file...\n\n");
	//save the residual image to file (optional?)
	save_image_to_file(residual, config.image_size, config.residual_output_image);

	// Clean up resources
	printf(">>> UPDATE: Cleaning up resources...\n\n");
	clean_up(&residual, &model, &psf);
	printf(">>> UPDATE: Clean up complete, exiting...\n\n");

	return EXIT_SUCCESS;
}