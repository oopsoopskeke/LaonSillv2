/*
 * DeepDream.cpp
 *
 *  Created on: 2016. 7. 16.
 *      Author: jhkim
 */

#include "DeepDream.h"

using namespace std;

#ifndef GPU_MODE
DeepDream::DeepDream(Network *network, const char *base_img, UINT iter_n, UINT octave_n,
		double octave_scale, const char *end, bool clip) {

	this->network = network;
	strcpy(this->base_img, base_img);
	this->iter_n = iter_n;
	this->octave_n = octave_n;
	this->octave_scale = octave_scale;
	strcpy(this->end, end);
	this->clip = clip;

	for(int i = 0; i < 3; i++) {
		mean[i] = network->getDataSetMean(i);
	}
}

DeepDream::~DeepDream() {}


void printImage(const char *head, DATATYPE *data, int w, int h, int c, bool force=false) {
	if(force || true) {
		int width = min(5, w);
		int height = min(5, h);

		cout << head << "-" << w << "x" << h << "x" << c << endl;
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {
				cout << data[i*w+j] << ", ";
			}
			cout << endl;
		}
	}
}


void printImage(const char *head, CImg<DATATYPE>& img, bool force=false) {
	if(force || true) {
		printImage(head, img.data(), img.width(), img.height(), img.spectrum(), force);
	}
}





void DeepDream::deepdream() {
	Util::setPrint(false);
	// prepare base images for all octaves
	CImg<DATATYPE> image(base_img);
	//Util::printData(image.data(), image.height(), image.width(), image.spectrum(), 1, "image:");
	image.normalize(0.0, 1.0);
	//Util::printData(image.data(), image.height(), image.width(), image.spectrum(), 1, "image:");
	preprocess(image);
	//Util::printData(image.data(), image.height(), image.width(), image.spectrum(), 1, "image:");
	CImgDisplay main_disp(image, "input image");

	vector<CImg<DATATYPE>> octaves(octave_n);
	for(int i = 0; i < octave_n; i++) {
		octaves[i] = image;
		cout << "image size for octave " << i << "-width: " << image.width() << ", height: " << image.height() << ", spectrum: " << image.spectrum() << endl;
		image.resize(image.width()/octave_scale, image.height()/octave_scale, -100, -100, 5);
	}

	CImg<DATATYPE> src;
	CImgDisplay process_disp(src, "proccess");
	// allocate image for network-produced details
	CImg<DATATYPE> detail(octaves[octave_n-1], "xyzc", 0.0);


	for(int octave_index = octave_n-1; octave_index >= 0; octave_index--) {
		CImg<DATATYPE>& octave_base = octaves[octave_index];
		Util::printData(octave_base.data(), octave_base.height(), octave_base.width(), octave_base.spectrum(), 1, "octave_base:");

		const int w = octave_base.width();
		const int h = octave_base.height();
		if(octave_index < octave_n-1) {
			// upscale details from the previous octave
			detail.resize(w, h, -100, -100, 5);
		}
		Util::printData(detail.data(), detail.height(), detail.width(), detail.spectrum(), 1, "detail:");


		// resize the network's input image size
		network->reshape(io_dim(h, w, octave_base.spectrum(), 1));
		src = octave_base + detail;

		DATATYPE *d_src;
		checkCudaErrors(cudaMalloc(&d_src, sizeof(DATATYPE)*src.size()));
		for(int i = 0; i < iter_n; i++) {
			make_step(src, d_src, end, 0.005);
			Util::printData(src.data(), src.height(), src.width(), src.spectrum(), 1, "src_after_make_step:");
			// reconstruction된 이미지를 다시 normalize ...
			//src.normalize(0.0, 1.0);
			//Util::printData(src.data(), src.height(), src.width(), src.spectrum(), 1, "src_after_make_step_normalize:");

			//visualization
			CImg<DATATYPE> temp_src(src);
			deprocess(temp_src);
			//temp_src.normalize(0.0, 1.0);

			// adjust image contrast if clipping is disabled
			if(!clip) {
				//vis = vis*(255.0/
			}
			process_disp.resize(temp_src, true).display(temp_src).wait(20);
			cout << "octave: " << octave_index << ", iter: " << i << ", end: " << end << ", dim: " << endl;
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////
		}
		checkCudaErrors(cudaFree(d_src));

		// extract details produced on the current octave
		detail = src - octave_base;
		Util::printData(src.data(), src.height(), src.width(), src.spectrum(), 1, "src:");
		Util::printData(octave_base.data(), octave_base.height(), octave_base.width(), octave_base.spectrum(), 1, "octave_base:");
		Util::printData(detail.data(), detail.height(), detail.width(), detail.spectrum(), 1, "detail:");
	}
	deprocess(src);

	char result_path[256];
	sprintf(result_path, "/home/jhkim/result-%s.jpg", end);
	src.normalize(0, 255);
	src.save_jpeg(result_path, 100);

	while(!process_disp.is_closed()) {
		process_disp.wait();
	}

	while(!main_disp.is_closed()) {
		main_disp.wait();
	}

}


void DeepDream::make_step(CImg<DATATYPE>& src, DATATYPE *d_src, const char *end, float step_size, int jitter) {
	//apply jitter shift
	srand((unsigned int)time(NULL));
	int ox = rand()%(jitter*2+1)-jitter;
	int oy = rand()%(jitter*2+1)-jitter;

	src.shift(ox, 0, 0, 0, 2);
	src.shift(0, oy, 0, 0, 2);

	DATATYPE* p_src = src.data();
	checkCudaErrors(cudaMemcpyAsync(d_src, p_src, sizeof(DATATYPE)*src.size(), cudaMemcpyHostToDevice));

	network->feedforward(d_src, end);
	Layer* dst = dynamic_cast<Layer*>(network->findLayer(end));
	if(!dst) {
		cout << "could not find layer of name " << end << " ... " << endl;
		exit(-1);
	}
	/*
	Layer* dst_nextLayer = dynamic_cast<Layer*>(dst->getNextLayers()[0]);
	if(!dst_nextLayer) {
		cout << "could not find next layer ... " << endl;
		exit(-1);
	}
	// specify the optimization objective
	dst_nextLayer->setDeltaInput(dst->getOutput());
	*/
	dst->backpropagation(0, dst->getOutput());
	Layer* firstHiddenLayer = dynamic_cast<Layer*>(network->getInputLayer()->getNextLayers()[0]);
	if(!firstHiddenLayer) {
		cout << "cout not find first hidden layer ... " << endl;
		exit(-1);
	}

	io_dim in_dim = network->getInputLayer()->getInDimension();
	int input_b_outsize = in_dim.batchsize();
	DATATYPE *g = new DATATYPE[input_b_outsize];
	checkCudaErrors(cudaMemcpyAsync(g, firstHiddenLayer->getDeltaInput(), sizeof(DATATYPE)*input_b_outsize, cudaMemcpyDeviceToHost));

	Util::printData(g, in_dim.rows, in_dim.cols, in_dim.channels, 1, "g:");
	Util::printData(p_src, in_dim.rows, in_dim.cols, in_dim.channels, 1, "src:");

	// apply normalizedascent step to the input image
	double g_sum = 0.0f;
	for(int i = 0; i < input_b_outsize; i++) {
		g_sum += abs(g[i]);
	}
	float g_coef = (float)(step_size/(g_sum / input_b_outsize));
	for(int i = 0; i < input_b_outsize; i++) {
		p_src[i] += g_coef*g[i];
	}

	//unshift image
	src.shift(-ox, 0, 0, 0, 2);
	src.shift(0, -oy, 0, 0, 2);

	//printImage("src", src, in_dim.cols, in_dim.rows, in_dim.channels, true);
	Util::printData(p_src, in_dim.rows, in_dim.cols, in_dim.channels, 1, "src:");

	if(clip) {
		clipImage(src);
	}

	delete [] g;
}





void DeepDream::objective_L2() {

}


void DeepDream::preprocess(CImg<DATATYPE>& img) {
	DATATYPE *data_ptr = img.data();
	const int height = img.height();
	const int width = img.width();
	const int channel = img.spectrum();

	for(int c = 0; c < channel; c++) {
		for(int h = 0; h < height; h++) {
			for(int w = 0; w < width; w++) {
				data_ptr[w+h*width+c*height*width] -= mean[c];
			}
		}
	}
}

void DeepDream::deprocess(CImg<DATATYPE>& img) {

	DATATYPE *data_ptr = img.data();
	const int height = img.height();
	const int width = img.width();
	const int channel = img.spectrum();

	for(int c = 0; c < channel; c++) {
		for(int h = 0; h < height; h++) {
			for(int w = 0; w < width; w++) {
				data_ptr[w+h*width+c*height*width] += mean[c];
			}
		}
	}
}

void DeepDream::clipImage(CImg<DATATYPE>& img) {
	DATATYPE *src_ptr = img.data();
	const int width = img.width();
	const int height = img.height();
	const int channel = img.spectrum();
	int index;
	for(int c = 0; c < channel; c++) {
		for(int h = 0; h < height; h++) {
			for(int w = 0; w < width; w++) {
				index = w+h*width+c*width*height;
				if(src_ptr[index] < -mean[c]) {
					src_ptr[index] = -mean[c];
				} else if(src_ptr[index] > 1.0-mean[c]) {
					src_ptr[index] = 1.0-mean[c];
				}
			}
		}
	}
}
#endif
