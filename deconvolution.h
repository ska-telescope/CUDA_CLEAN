
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

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DECONVOLUTION_H_
#define DECONVOLUTION_H_

	// Global toggle switch for single
	// or double precision calculations
	#define SINGLE_PRECISION 0

	#if SINGLE_PRECISION
		#define PRECISION float
		#define PRECISION2 float2
		#define PRECISION3 float3
		#define PRECISION4 float4
	#else
		#define PRECISION double
		#define PRECISION2 double2
		#define PRECISION3 double3
		#define PRECISION4 double4
	#endif

	#if SINGLE_PRECISION
		#define SIN(x) sinf(x)
		#define COS(x) cosf(x)
		#define ABS(x) fabsf(x)
		#define SQRT(x) sqrtf(x)
		#define ROUND(x) roundf(x)
		#define CEIL(x) ceilf(x)
		#define FLOOR(x) floorf(x)
		#define MAKE_PRECISION2(x,y) make_float2(x,y)
		#define MAKE_PRECISION3(x,y,z) make_float3(x,y,z)
		#define MAKE_PRECISION4(x,y,z,w) make_float4(x,y,z,w)
	#else
		#define SIN(x) sin(x)
		#define COS(x) cos(x)
		#define ABS(x) fabs(x)
		#define SQRT(x) sqrt(x)
		#define ROUND(x) round(x)
		#define FLOOR(x) floor(x)
		#define CEIL(x) ceil(x)
		#define MAKE_PRECISION2(x,y) make_double2(x,y)
		#define MAKE_PRECISION3(x,y,z) make_double3(x,y,z)
		#define MAKE_PRECISION4(x,y,z,w) make_double4(x,y,z,w)
	#endif

	typedef struct Config {
		unsigned int image_size;
		unsigned int psf_size;
		unsigned int number_minor_cycles;
		// unsigned int maximum_sources_per_cycle;
		double loop_gain;
		char *dirty_real_file;
		char *dirty_imag_file;
		char *residual_real_file;
		char *residual_imag_file;
		char *model_real_file;
		char *model_imag_file;
		char *psf_real_file;
		char *psf_imag_file;
	} Config;

	typedef struct Complex {
		PRECISION real;
		PRECISION imag;
	} Complex;

	typedef struct Source {
		PRECISION l;
		PRECISION m;
		PRECISION intensity;
	} Source;

#endif /* DECONVOLUTION_H_ */

#ifdef __cplusplus
}
#endif