
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

#include "deconvolution.h"

int main(int argc, char **argv)
{
	// Initialise the config
	Config config;
	init_config(&config);

	// Allocate memory for images (residual/model), and PSF kernel
	Complex *residual = NULL;
	Complex *model = NULL;
	Complex *psf = NULL;
	allocate_image_resources(&residual, &model, &psf, config.image_size, config.psf_size);

	if(residual == NULL || model == NULL || psf == NULL)
	{
		printf(">>> ERROR: Unable to allocate required resources, terminating...\n\n");
		clean_up(&residual, &model, &psf);
		return EXIT_FAILURE;
	}

	// Load dirty image into memory as residual
	bool loaded_dirty = load_image_from_file(residual, config.dirty_real_file, 
		config.dirty_imag_file, config.image_size);
	// Load point spread function kernel into memory
	bool loaded_psf = load_image_from_file(psf, config.psf_real_file, 
		config.psf_imag_file, config.psf_size);

	if(!loaded_dirty || !loaded_psf)
	{
		printf(">>> ERROR: Unable to load dirty or psf image from file, terminating...\n\n");
		clean_up(&residual, &model, &psf);
		return EXIT_FAILURE;
	}

	// Perform CLEAN
	performing_deconvolution(&config, residual, model, psf);

	// Save model and residual images to file
	save_image_to_file(model, config.model_real_file, config.model_imag_file, config.image_size);
	save_image_to_file(residual, config.residual_real_file, config.residual_imag_file, config.image_size);

	// Clean up resources
	clean_up(&residual, &model, &psf);

	return EXIT_SUCCESS;
}