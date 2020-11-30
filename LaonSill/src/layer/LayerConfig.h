/**
 * @file LayerConfig.h
 * @date 2016/5/14
 * @author jhkim
 * @brief 레이어 설정과 관련된 열거형, 구조체 등을 선언
 * @details
 */

#ifndef LAYERCONFIG_H_
#define LAYERCONFIG_H_

#include <cmath>
#include <chrono>
#include <random>

#include <boost/random.hpp>
#include <boost/math/special_functions/next.hpp>

#include "common.h"
#include "Util.h"
#include "Data.h"
#include "EnumDef.h"
#include "MathFunctions.h"



//typedef arma::fvec rvec;
//typedef arma::fmat rmat;
//typedef arma::fcube rcube;
//typedef unsigned int UINT;

typedef boost::mt19937 RNGType;

/**
 * @brief 데이터의 구조정보를 정의하는 구조체
 */
struct io_dim {
    UINT rows;				///< 데이터의 행 수, 일반 1차원 데이터의 경우 한 데이터의 유효 
                            ///  엘리먼트의 수, 이미지와 같은 3차원 데이터의 경우 이미지의 
                            //   height값을 따른다.
    UINT cols;				///< 데이터의 열 수, 일반 1차원 데이터의 경우 1, 이미지와 같은 
                            ///  3차원 데이터의 경우 이미지의 width값을 따른다.
    UINT channels;			///< 데이터의 채널 수, 일반 1차원 데이터의 경우 1, 이미지와 같은 
                            ///  3차원 데이터의 경우 RGB채널이 있는 경우 3, GrayScale인 경우 
                            ///  1의 값이 된다.
    UINT batches;	        ///< 한 번에 학습하는 데이터의 수를 의미한다.

    io_dim() {

    }
    io_dim(UINT rows, UINT cols, UINT channels, UINT batches) {
    	this->rows = rows;
    	this->cols = cols;
    	this->channels = channels;
    	this->batches = batches;
    }
    //int size() const { return rows*cols*channels*batches; }

    void print() {
    	std::cout << "rows: " << this->rows << std::endl;
    	std::cout << "cols: " << this->cols << std::endl;
    	std::cout << "channels: " << this->channels << std::endl;
    	std::cout << "batches: " << this->batches << std::endl;
    }

    /**
     * @details 데이터 하나의 엘리먼트 크기를 조회한다.
     * @return 데이터 하나의 엘리먼트 크기
     */
    int unitsize() const { return rows*cols*channels; }
    /**
     * @details 배치 단위의 데이터 엘리먼트 크기를 조회한다.
     * @return 배치 단위의 데이터 엘리먼트 크기
     */
    int batchsize() const { return rows*cols*channels*batches; }
};


/**
 * @brief 컨볼루션 연산을 정의하는 파라미터 구조체
 * @todo io_dim의 구조를 따르는 면이 있어(row, colum, channel...) 상속받았으나 batch값은 
 *       적합하지 않음.
 *       상속받지 않고 별도의 필드를 정의하는 것이 바람직해 보인다.
 */
struct filter_dim : public io_dim {
	uint32_t filters;			///< 컨볼루션 결과 출력 채널(필터)의 수
	uint32_t pad;
	uint32_t pad_h;
	uint32_t pad_w;
	uint32_t stride;			///< 컨볼루션을 적용할 stride 크기
	uint32_t dilation;

	filter_dim() {}
	filter_dim(uint32_t rows, uint32_t cols, uint32_t channels, uint32_t filters,
			uint32_t pad, uint32_t pad_h, uint32_t pad_w, uint32_t stride, uint32_t dilation)
	: io_dim(rows, cols, channels, 1) {
		this->filters = filters;
		this->pad = pad;
		this->pad_h = pad_h;
		this->pad_w = pad_w;
		this->stride = stride;
		this->dilation = dilation;
	}

	void print() {
		io_dim::print();

		std::cout << "filters: " << this->filters << std::endl;
		std::cout << "pad: " << this->pad << std::endl;
		std::cout << "pad_h: " << this->pad_h << std::endl;
		std::cout << "pad_w: " << this->pad_w << std::endl;
		std::cout << "stride: " << this->stride << std::endl;
		std::cout << "dilation: " << this->dilation << std::endl;
	}

	/**
	 * @details 전체 필터의 엘리먼트 크기를 조회한다.
	 * @return 전체 필터의 엘리먼트 크기
	 */
	int size() const {
		return rows*cols*channels*filters;
	}
};

/**
 * @brief 풀링 연산을 정의하는 파라미터 구조체
 * @todo padding 적용을 위한 필드를 추가해야 함
 */
struct pool_dim {
	uint32_t rows;				///< 풀링 커널의 height 값
	uint32_t cols;				///< 풀링 커널의 width 값
	uint32_t pad;
	uint32_t stride;			///< 풀링을 적용할 stride 크기

	//pool_dim(uint32_t rows=1, uint32_t cols=1, uint32_t stride=1) {
	pool_dim() {}
	pool_dim(uint32_t rows, uint32_t cols, uint32_t pad, uint32_t stride) {
		this->rows = rows;
		this->cols = cols;
		this->pad = pad;
		this->stride = stride;
	}

	void print() {
		std::cout << "rows: " << this->rows << std::endl;
		std::cout << "cols: " << this->cols << std::endl;
		std::cout << "pad: " << this->pad << std::endl;
		std::cout << "stride: " << this->stride << std::endl;
	}
};



/**
 * @brief LRN 연산을 정의하는 파라미터 구조체
 * @details 각 파라미터는 (1+(α/n)∑ixi^2)^β 수식을 참고한다.
 */
struct lrn_dim {
	UINT local_size;		///< 정규화를 적용할 채널의 수
	double alpha;			///< 스케일링 파라미터
	double beta;			///< 지수
	double k;

	//lrn_dim(UINT local_size=5, double alpha=1, double beta=5) {
	lrn_dim(UINT local_size=5, double alpha=1.0, double beta=0.75, double k=1.0) {
		this->local_size = local_size;
		this->alpha = alpha;
		this->beta = beta;
		this->k = k;
	}
};

/**
 * @brief 학습 파라미터 업데이트 파라미터 구조체
 */
struct update_param {
	double lr_mult;			///< learning rate
	double decay_mult;		///< weight decay

	update_param() : update_param(1.0, 1.0) {}
	update_param(double lr_mult, double decay_mult) {
		this->lr_mult = lr_mult;
		this->decay_mult = decay_mult;
	}

	void print() {
		std::cout << "lr_mult: " << this->lr_mult << std::endl;
		std::cout << "decay_mult: " << this->decay_mult << std::endl;
	}
};

/**
 * @brief 학습 파라미터 초기화 파라미터 구조체
 */
template <typename Dtype>
struct param_filler {
	ParamFillerType type;	///< 파라미터 초기화 타입
	float value;			///< 파라미터 초기화 관련 값
	float mean;
	float std;
	RNGType rng;

	param_filler() {}
	param_filler(ParamFillerType type, float value = 0.f, float mean = 0.f, float std = 1.f) {
		this->type = type;
		this->value = value;
		this->mean = mean;
		this->std = std;
		this->rng.seed(static_cast<unsigned int>(time(NULL)+getpid()));
	}

	void print() {
		std::cout << "type: " << this->type << std::endl;
		std::cout << "value: " << this->value << std::endl;
		std::cout << "mean: " << this->mean << std::endl;
		std::cout << "std: " << this->std << std::endl;
	}

	/**
	 * @details 학습 파라미터를 초기화한다.
	 * @param param 파라미터 장치 메모리 포인터
	 * @param size 파라미터 사이즈
	 * @param n_in 레이어의 입력 노드 수
	 * @param n_out 레이어의 출력 노드 수
	 */
	//void fill(DATATYPE *param, int size, int n_in, int n_out) {
	void fill(Data<Dtype>* data) {
		Dtype* mem = data->mutable_host_data();
		switch(type) {
		case ParamFillerType::Constant: {
			size_t size = data->getCount();
			for(uint32_t i = 0; i < size; i++) mem[i] = this->value;
			break;
		}
		case ParamFillerType::Xavier: {
			/*
			size_t size = data->getCount();
			int fan_in = size / data->batches();
			int fan_out = size / data->channels();
			Dtype n = fan_in;
			Dtype scale = sqrt(Dtype(3)/n);

			// BOOST
			boost::uniform_real<float> random_distribution(-scale,
                boost::math::nextafter<float>(scale, std::numeric_limits<float>::max()));
			boost::variate_generator<RNGType, boost::uniform_real<float> > 
                variate_generator(rng, random_distribution);

			for(uint32_t i = 0; i < size; ++i) {
				mem[i] = variate_generator();
			}
			*/

			SASSERT0(data->getCount());
			int fan_in = data->getCount() / data->batches();
			int fan_out = data->getCount() / data->channels();
			Dtype n = fan_in;		// default to fan_in

			// XXX: FAN_OUT, AVERAGE Option 적용되지 않음
			const float scale = std::sqrt(3.f / n);
			//std::cout << "fan_in: " << fan_in << std::endl;
			//std::cout << "fan_out: " << fan_out << std::endl;
			//std::cout << "scale: " << scale << std::endl;


			soooa_rng_uniform<Dtype>(data->getCount(), -scale, scale,
					data->mutable_host_data());
			break;
		}
		case ParamFillerType::Gaussian: {
			/*
			size_t size = data->getCount();
			boost::normal_distribution<float> random_distribution(0, value);
			boost::variate_generator<RNGType, boost::normal_distribution<float> >
			variate_generator(rng, random_distribution);
			for (uint32_t i = 0; i < size; ++i) {
				mem[i] = variate_generator();
			}
			*/
			SASSERT0(data->getCount());
			soooa_rng_gaussian<Dtype>(data->getCount(), this->mean, this->std,
					data->mutable_host_data());

			//XXX: sparse 무시
			break;
		}
		case ParamFillerType::MSRA: {
			SASSERT0(data->getCount());
			int fan_in = data->getCount() / data->batches();
			int fan_out = data->getCount() / data->channels();
			Dtype n = fan_in;		// default to fan_in

			// XXX: FAN_OUT, AVERAGE Option 적용되지 않음
			const float std = std::sqrt(2.f / n);

			//std::cout << "fan_in: " << fan_in << std::endl;
			//std::cout << "fan_out: " << fan_out << std::endl;
			//std::cout << "std: " << std << std::endl;

			soooa_rng_gaussian<Dtype>(data->getCount(), 0.0f, std,
					data->mutable_host_data());
			break;
		}
		default:
			break;
		}

		//print();
		//Data<Dtype>::printConfig = 1;
		//SyncMem<Dtype>::printConfig = 1;
		//data->print_data({}, false);
		//Data<Dtype>::printConfig = 0;
		//SyncMem<Dtype>::printConfig = 0;
	}

private:


};


/*
struct next_layer_relation {
	Layer *next_layer;
	UINT idx;

	next_layer_relation() {}
	next_layer_relation(Layer *next_layer, UINT idx) {
		this->next_layer = next_layer;
		this->idx = idx;
	}
};

struct prev_layer_relation {
	Layer *prev_layer;
	UINT idx;

	prev_layer_relation() {}
	prev_layer_relation(Layer *prev_layer, UINT idx) {
		this->prev_layer = prev_layer;
		this->idx = idx;
	}
};
*/

#endif /* LAYERCONFIG_H_ */
