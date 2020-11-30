/*
 * DeepDream.h
 *
 *  Created on: 2016. 7. 16.
 *      Author: jhkim
 */

#ifndef DEEPDREAM_H_
#define DEEPDREAM_H_

#ifndef GPU_MODE
#include <CImg.h>

#include "common.h"
#include "Network.h"

using namespace cimg_library;

class DeepDream {
public:
	DeepDream(Network *network, const char *base_img, UINT iter_n=10, UINT octave_n=4,
			double octave_scale=1.4, const char *end="inception_4c/output", bool clip=true);
	virtual ~DeepDream();

	void deepdream();

private:
	void make_step(CImg<DATATYPE>& src, DATATYPE* d_src, const char* end, float step_size=1.5, int jitter=32);
	void objective_L2();
	void preprocess(CImg<DATATYPE>& img);
	void deprocess(CImg<DATATYPE>& img);
	void clipImage(CImg<DATATYPE>& img);

	Network *network;
	char base_img[256];
	UINT iter_n;
	UINT octave_n;
	double octave_scale;
	char end[256];
	bool clip;

	DATATYPE mean[3];
};
#endif


#endif /* DEEPDREAM_H_ */
