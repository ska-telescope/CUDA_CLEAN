
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

void init_config(Config *config)
{
	config->image_size = 1024;

	config->psf_size = 9; // 9x9

	config->number_minor_cycles = 60;

	config->loop_gain = 0.1; // 0.1 is typical

	config->dirty_real_file = "../images/dirty_real.csv";
	config->dirty_imag_file = "../images/dirty_imag.csv";
	config->model_real_file = "../images/model_real.csv";
	config->model_imag_file = "../images/model_imag.csv";
	config->psf_real_file = "../images/psf_real.csv";
	config->psf_imag_file = "../images/psf_imag.csv";
}

void allocate_image_resources(Complex **residual, Complex **model, Complex **psf,
	unsigned int image_size, unsigned int psf_size)
{
	unsigned int image_size_squared = image_size * image_size;
	*residual = (Complex*) calloc(image_size_squared, sizeof(Complex));
	*model = (Complex*) calloc(image_size_squared, sizeof(Complex));
	*psf = (Complex*) calloc(psf_size * psf_size, sizeof(Complex));
}

void performing_deconvolution(Config *config, Complex *residual, Complex *model, Complex *psf)
{
	// Get coordinates and absolute value of maximum peak value in PSF

	// Normalize PSF by its absolute maximum value

	// Normalize residual image by PSF absolute maximum value

	// More to follow...
}

bool load_image_from_file(Complex *image, unsigned int size, char *real_file, char *imag_file)
{
	FILE *real_f = fopen(real_file, "r");
	FILE *imag_f = fopen(imag_file, "r");

	if(real_f == NULL || imag_f == NULL)
	{
		printf(">>> ERROR: Unable to load image from file...\n\n");
		if(real_f) fclose(real_f);
		if(imag_f) fclose(imag_f);
		return false;
	}

	for(int row = 0; row < size; ++row)
	{
		for(int col = 0; col < size; ++col)
		{
			unsigned int image_index = row * size + col;

			#if SINGLE_PRECISION
				fscanf(real_f, "%f ", &(image[image_index].real));
				fscanf(imag_f, "%f ", &(image[image_index].imag));
			#else
				fscanf(real_f, "%lf ", &(image[image_index].real));
				fscanf(imag_f, "%lf ", &(image[image_index].imag));
			#endif
		}
	}

	fclose(real_f);
	fclose(imag_f);
	return true;
}

void save_image_to_file(Complex *image, unsigned int size, char *real_file, char *imag_file)
{
	FILE *real_f = fopen(real_file, "w");
	FILE *imag_f = fopen(imag_file, "w");

	if(real_f == NULL || imag_f == NULL)
	{
		printf(">>> ERROR: Unable to save image to file, moving on...\n\n");
		if(real_f) fclose(real_f);
		if(imag_f) fclose(imag_f);
		return;
	}

	for(int row = 0; row < size; ++row)
	{
		for(int col = 0; col < size; ++col)
		{
			unsigned int image_index = row * size + col;

			#if SINGLE_PRECISION
				fprintf(real_f, "%f ", image[image_index].real);
				fprintf(imag_f, "%f ", image[image_index].imag);
			#else
				fprintf(real_f, "%lf ", image[image_index].real);
				fprintf(imag_f, "%lf ", image[image_index].imag);
			#endif
		}

		fprintf(real_f, "\n");
		fprintf(imag_f, "\n");
	}

	fclose(real_f);
	fclose(imag_f);
}

void clean_up(Complex **residual, Complex **model, Complex **psf)
{
	if(*residual) free(*residual);
	if(*model)    free(*model);
	if(*psf)      free(*psf);
}