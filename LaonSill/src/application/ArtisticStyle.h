/*
 * ArtisticStyle.h
 *
 *  Created on: Mar 17, 2017
 *      Author: jkim
 */
#if 0

#ifndef ARTISTICSTYLE_H_
#define ARTISTICSTYLE_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <map>

#include "Network.h"


template <typename Dtype>
class ArtisticStyle {
public:
	ArtisticStyle();
	virtual ~ArtisticStyle();

	void transfer_style();


private:
	void load_model();
	void rescale_net(const std::vector<uint32_t>& shape);

	/**
	 * @brief Computes representation matrices for an image
	 */
	std::vector<std::map<std::string, Data<Dtype>*>> compute_reprs(Data<Dtype>* net_in,
			const std::vector<std::string>& layers_style,
			const std::vector<std::string>& layers_content, const Dtype gram_scale = 1.0f);
	/**
	 * @brief Computes style gradient and loss from activation features.
	 */
	Dtype compute_style_grad(std::map<std::string, Data<Dtype>*>& F,
			std::map<std::string, Data<Dtype>*>& G, const std::string& layer,
			Data<Dtype>* grad);
	/**
	 * @brief Computes content gradient and loss from activation features.
	 */
	Dtype compute_content_grad(std::map<std::string, Data<Dtype>*>& F,
			const std::string& layer, Data<Dtype>* grad);

	void transformer_preprocess(Data<Dtype>* data);
	void transformer_deprocess(Data<Dtype>* data);

	void net_forward(Data<Dtype>* net_in);
	/**
	 * Style transfer optimization callback
	 */
	Dtype style_optfn(Data<Dtype>* net_in);



	void _on();
	void _off();

	Data<Dtype>* _from_mat_to_data(const cv::Mat& mat);
	uint32_t _max_from_shape(const std::vector<uint32_t>& shape, int from, int to);
	uint32_t _min_from_shape(const std::vector<uint32_t>& shape, int from, int to);
	uint32_t _max_dim_from_mat(const cv::Mat& mat);
	uint32_t _min_dim_from_mat(const cv::Mat& mat);

	const std::vector<std::string> _map_keys(std::map<std::string, Dtype>& arg);

	Data<Dtype>* _generateInitialInput();

	void _clearNameDataMapVector(std::vector<std::map<std::string, Data<Dtype>*>>& v);
	void _clearNameDataMap(std::map<std::string, Data<Dtype>*>& m);

private:
	Network<Dtype>* network;

	std::string		end;
	std::string 	style_img;
	std::string 	content_img;
	int 			length;
	Dtype 			ratio;
	int 			n_iter;
	int 			init;
	std::string		content_type;

	// Update Params
	std::string optimizer_type;
	Dtype lr;
	Dtype wd;
	Dtype mt;
	Dtype eps;
	Dtype bt1;
	Dtype bt2;
	Data<Dtype>* hist;
	Data<Dtype>* hist2;

	cv::Mat 		cv_img_style;
	cv::Mat 		cv_img_content;
	Data<Dtype>* 	img_style;
	Data<Dtype>* 	img_content;
	std::map<std::string, Data<Dtype>*>	G_style;
	std::map<std::string, Data<Dtype>*>	F_content;


	std::map<std::string, std::map<std::string, Dtype>> weights;
	std::vector<std::string> layers;

	std::vector<Layer<Dtype>*> _layers;
	std::map<std::string, Data<Dtype>*> _layerDataMap;
	std::map<std::string, Layer<Dtype>*> _nameLayerMap;



	Data<Dtype>*	mean;
	Data<Dtype>*	data_bounds;


};

#endif /* ARTISTICSTYLE_H_ */


#endif
