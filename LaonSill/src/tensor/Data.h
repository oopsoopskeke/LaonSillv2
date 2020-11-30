/*
 * Data.h
 *
 *  Created on: 2016. 8. 19.
 *      Author: jhkim
 */

#ifndef DATA_H_
#define DATA_H_

#include <cstdint>
#include <vector>
#include <memory>

#include "common.h"
#include "Util.h"
#include "SyncMem.h"
#include "SysLog.h"

/**
 * @brief Layer 특정 단계에서의 data, gradient를 pair로 warpping, util method를 제공하는
 *       클래스
 * @details Layer의 입력, 출력, Parameter에 해당하는 data와 gradient에 적용
 */
template <typename Dtype>
class Data {
public:
	//Data(const bool hostOnly=false);
	Data(const std::string& name = "", const bool hostOnly=false);
	Data(Data<Dtype>* data, const bool hostOnly=false);
	Data(const std::string& name, Data<Dtype>* data, uint32_t type,
        const bool hostOnly=false);
	Data(const std::string& name, const std::vector<uint32_t>& shape,
        const bool hostOnly=false);
	virtual ~Data();

	//void shape(const std::vector<uint32_t>& shape);
	void reshape(const std::vector<uint32_t>& shape);
	void reshapeInfer(const std::vector<int>& shape);
	void reshapeLike(const Data<Dtype>* data);


	size_t getCount() const { return _count; }
	size_t getCountByAxis(uint32_t axis, uint32_t end=0) const {
		if (end == 0) end = _shape.size();

		size_t count = 1;
		for (uint32_t i = axis; i < end; i++) {
			count *= _shape[i];
		}
		return count;
	}
	uint32_t getShape(uint32_t axis) { return _shape[axis]; }
	const std::vector<uint32_t>& getShape() const { return _shape; }


	/**
	 * @details 데이터의 수정할 수 없는 호스트 메모리 포인터 조회한다.
	 * @return 데이터의 수정할 수 없는 호스트 메모리 포인터
	 */
	const Dtype* host_data();
	/**
	 * @details 데이터의 수정할 수 없는 디바이스 메모리 포인터 조회한다.
	 * @return 데이터의 수정할 수 없는 디바이스 메모리 포인터
	 */
	const Dtype* device_data();
	/**
	 * @details 그레디언트의 수정할 수 없는 호스트 메모리 포인터 조회한다.
	 * @return 그레디언트의 수정할 수 없는 호스트 메모리 포인터
	 */
	const Dtype* host_grad();
	/**
	 * @details 그레디언트의 수정할 수 없는 디바이스 메모리 포인터 조회한다.
	 * @return 그레디언트의 수정할 수 없는 디바이스 메모리 포인터
	 */
	const Dtype* device_grad();

	/**
	 * @details 데이터의 수정할 수 있는 호스트 메모리 포인터 조회한다.
	 * @return 데이터의 수정할 수 있는 호스트 메모리 포인터
	 */
	Dtype* mutable_host_data();
	/**
	 * @details 데이터의 수정할 수 있는 디바이스 메모리 포인터 조회한다.
	 * @return 데이터의 수정할 수 있는 디바이스 메모리 포인터
	 */
	Dtype* mutable_device_data();
	/**
	 * @details 그레디언트의 수정할 수 있는 호스트 메모리 포인터 조회한다.
	 * @return 그레디언트의 수정할 수 있는 호스트 메모리 포인터
	 */
	Dtype* mutable_host_grad();
	/**
	 * @details 그레디언트의 수정할 수 있는 디바이스 메모리 포인터 조회한다.
	 * @return 그레디언트의 수정할 수 있는 디바이스 메모리 포인터
	 */
	Dtype* mutable_device_grad();


	void set(Data<Dtype>* data, bool reshape=true, int type = 0);

	/**
	 * @details 데이터의 호스트 메모리에 _count만큼 값을 복사한다.
	 * @param data 복사할 Data
	 * @param offset data의 포인터에 대한 offset
	 */
	void set_host_data(Data* data, const uint32_t offset=0, bool reshape = false) {
		if (reshape)
			this->reshapeLike(data);

        set_host_data(data->host_data()+offset);
    }
	/**
	 * @details 데이터의 디바이스 메모리에 _count만큼 값을 복사한다.
	 * @param data 복사할 Data
	 * @param offset data의 포인터에 대한 offset
	 */
	void set_device_data(Data* data, const uint32_t offset=0, bool reshape = false) {
		if (reshape)
			this->reshapeLike(data);

        set_device_data(data->device_data()+offset);
    }
	/**
	 * @details 그레디언트의 호스트 메모리에 _count만큼 값을 복사한다.
	 * @param grad 복사할 Data
	 * @param offset grad의 포인터에 대한 offset
	 */
	void set_host_grad(Data* grad, const uint32_t offset=0, bool reshape = false) {
		if (reshape)
			this->reshapeLike(grad);

        set_host_grad(grad->host_grad()+offset);
    }
	/**
	 * @details 그레디언트의 디바이스 메모리에 _count만큼 값을 복사한다.
	 * @param grad 복사할 Data
	 * @param offset grad의 포인터에 대한 offset
	 */
	void set_device_grad(Data* grad, const uint32_t offset=0, bool reshape = false) {
		if (reshape)
			this->reshapeLike(grad);

        set_device_grad(grad->device_grad()+offset);
    }
	/**
	 * @details 데이터의 호스트 메모리에 _count만큼 값을 복사한다.
	 * @param data 복사할 로우 데이터의 포인터
	 */
	void set_host_data(const Dtype* data, const uint32_t offset = 0);
	/**
	 * @details 데이터의 호스트 메모리에 _count만큼 주어진 디바이스 메모리 포인터로부터 값을
     *         복사한다.
	 * @param data 복사할 로우 디바이스 데이터의 포인터
	 */
	void set_host_with_device_data(const Dtype* data);
	/**
	 * @details 데이터의 디바이스 메모리에 _count만큼 주어진 호스트 메모리 포인터로부터 값을
     *         복사한다.
	 * @param data 복사할 로우 호스트 데이터의 포인터
	 */
	void set_device_with_host_data(const Dtype* data, const size_t offset=0,
        const size_t size=0);
	/**
	 * @details 데이터의 디바이스 메모리에 _count만큼 값을 복사한다.
	 * @param data 복사할 로우 디바이스 데이터의 포인터
	 */
	void set_device_data(const Dtype* data);

	/**
	 * @details 그레디언트의 호스트 메모리에 _count만큼 값을 복사한다.
	 * @param data 복사할 로우 호스트 데이터의 포인터
	 */
	void set_host_grad(const Dtype* grad);
	/**
	 * @details 그레디언트의 디바이스 메모리에 _count만큼 값을 복사한다.
	 * @param data 복사할 로우 디바이스 데이터의 포인터
	 */
	void set_device_grad(const Dtype* grad);

	/**
	 * @details 데이터의 호스트 메모리를 0으로 초기화한다.
	 */
	void reset_host_data(const bool setZero=true, const Dtype value=0.0);
	/**
	 * @details 데이터의 디바이스 메모리를 0으로 초기화한다.
	 */
	void reset_device_data(const bool setZero=true, const Dtype value=0.0);
	/**
	 * @details 그레디언트의 호스트 메모리를 0으로 초기화한다.
	 */
	void reset_host_grad();
	/**
	 * @details 그레디언트의 디바이스 메모리를 0으로 초기화한다.
	 */
	void reset_device_grad();


	/**
	 * @details 데이터의 호스트 메모리에 주어진 Data의 값을 더한다.
	 * @param data 더 할 Data
	 * @param offset data의 포인터에 대한 offset
	 */
	void add_host_data(Data* data, const uint32_t offset=0) {
        add_host_data(data->host_data()+offset);
    }
	/**
	 * @details 데이터의 디바이스 메모리에 주어진 Data의 값을 더한다.
	 * @param data 더 할 Data
	 * @param offset data의 포인터에 대한 offset
	 */
	void add_device_data(Data* data, const uint32_t offset=0) {
        add_device_data(data->device_data()+offset);
    }
	/**
	 * @details 그레디언트의 호스트 메모리에 주어진 Data의 값을 더한다.
	 * @param grad 더 할 Data
	 * @param offset data의 포인터에 대한 offset
	 */
	void add_host_grad(Data* grad, const uint32_t offset=0) {
        add_host_grad(grad->host_grad()+offset);
    }
	/**
	 * @details 그레디언트의 디바이스 메모리에 주어진 Data의 값을 더한다.
	 * @param grad 더 할 Data
	 * @param offset data의 포인터에 대한 offset
	 */
	void add_device_grad(Data* grad, const uint32_t offset=0) {
        add_device_grad(grad->device_grad()+offset);
    }

	void sub_host_data(Data* data) {
		_data->sub_host_mem(data->_data->host_mem());
	}
	void sub_device_data(Data* data) {
		_data->sub_device_mem(data->_data->device_mem());
	}
	void sub_host_grad(Data* data) {
		_grad->sub_host_mem(data->_grad->host_mem());
	}
	void sub_device_grad(Data* data) {
		_grad->sub_device_mem(data->_grad->device_mem());
	}
	/**
	 * @details 데이터의 호스트 메모리에 주어진 로우 호스트 포인터의 메모리 값을 더한다.
	 * @param data 더 할 로우 호스트 메모리 포인터
	 */
	void add_host_data(const Dtype* data);
	/**
	 * @details 데이터의 디바이스 메모리에 주어진 로우 디바이스 포인터의 메모리 값을 더한다.
	 * @param data 더 할 로우 디바이스 메모리 포인터
	 */
	void add_device_data(const Dtype* data);
	/**
	 * @details 그레디언트의 호스트 메모리에 주어진 로우 호스트 포인터의 메모리 값을 더한다.
	 * @param grad 더 할 로우 호스트 메모리 포인터
	 */
	void add_host_grad(const Dtype* grad);
	/**
	 * @details 그레디언트의 디바이스 메모리에 주어진 로우 디바이스 포인터의 메모리 값을
     *         더한다.
	 * @param grad 더 할 로우 디바이스 메모리 포인터
	 */
	void add_device_grad(const Dtype* grad);

	/**
	 * @details 데이터의 호스트 메모리를 스케일링한다.
	 * @param scale 스케일링할 스케일
	 */
	void scale_host_data(const float scale);
	/**
	 * @details 데이터의 디바이스 메모리를 스케일링한다.
	 * @param scale 스케일링할 스케일
	 */
	void scale_device_data(const float scale);
	/**
	 * @details 그레디언트의 호스트 메모리를 스케일링한다.
	 * @param scale 스케일링할 스케일
	 */
	void scale_host_grad(const float scale);
	/**
	 * @details 그레디언트의 디바이스 메모리를 스케일링한다.
	 * @param scale 스케일링할 스케일
	 */
	void scale_device_grad(const float scale);

	/**
	 * @details 데이터의 디바이스 메모리의 제곱합을 구한다.
	 * @param 데이터의 디바이스 메모리 제곱합
	 */
	double sumsq_device_data();
	/**
	 * @details 그레디언트의 디바이스 메모리의 제곱합을 구한다.
	 * @param 그레디언트의 디바이스 메모리 제곱합
	 */
	double sumsq_device_grad();

	double asum_device_data();
	double asum_device_grad();

	/**
	 * @details Data의 batch shape를 조회한다.
	 * @return Data의 batch shape
	 */
	inline uint32_t batches() const { return _shape[0]; }
	/**
	 * @details Data의 channel shape를 조회한다.
	 * @return Data의 channel shape
	 */
	inline uint32_t channels() const { return _shape[1]; }
	/**
	 * @details Data의 height shape를 조회한다.
	 * @return Data의 height shaper
	 */
	inline uint32_t height() const { return _shape[2]; }
	/**
	 * @details Data의 width shape를 조회한다.
	 * @return Data의 width shape
	 */
	inline uint32_t width() const { return _shape[3]; }

	inline uint32_t numAxes() const { return _shape.size(); }

	inline int offset(const int n, const int c = 0, const int h = 0, const int w = 0) const {
		SASSERT0(n >= 0);
		SASSERT0(n <= batches());
		SASSERT0(c >= 0);
		SASSERT0(c <= channels());
		SASSERT0(h >= 0);
		SASSERT0(h <= height());
		SASSERT0(w >= 0);
		SASSERT0(w <= width());
		return ((n * channels() + c) * height() + h) * width() + w;
	}




	bool is_nan_data() { return _data->is_nan_mem(); }
	bool is_nan_grad() { return _grad->is_nan_mem(); }
	bool is_inf_data() { return _data->is_inf_mem(); }
	bool is_inf_grad() { return _grad->is_inf_mem(); }

	uint32_t bound_data() { return _data->bound_mem(); }
	uint32_t bound_grad() { return _grad->bound_mem(); }

	void share_data(Data<Dtype>* data);
	void share_grad(Data<Dtype>* data);

	void save(const std::string& filename);
	void save(std::ofstream& ofs);
	void load(const std::string& filename);
	void load(std::ifstream& ifs);

	void print();
	/**
	 * @details 데이터를 shape에 따라 화면에 출력한다.
	 * @param head 출력할 때 헤드에 쓰일 문구
	 */
	// cmo == true 데이터가 메모리에 물리적으로 cmo로 저장되어 있는데 일반 행렬의 순(rmo)로 출력하는 경우
	void print_data(const std::string& head, const std::vector<uint32_t>& shape = {},
			const bool cmo=true, const int summary = 6);
	void print_data(const std::vector<uint32_t>& shape = {}, const bool cmo=true,
			const int summary = 6);
	void print_data_flatten();

	void print_shape();

	/**
	 * @details 그레디언트를 shape에 따라 화면에 출력한다.
	 * @param head 출력할 때 헤드에 쓰일 문구
	 */
	void print_grad(const std::string& head, const std::vector<uint32_t>& shape = {},
			const bool cmo=true, const int summary = 6);
	void print_grad(const std::vector<uint32_t>& shape = {}, const bool cmo=true,
			const int summary = 6);





	void fill_host_with_1d_vec(const std::vector<int>& array,
			const std::vector<uint32_t>& transpose={0, 1, 2, 3});
	void fill_host_with_1d_vec(const std::vector<uint32_t>& array,
			const std::vector<uint32_t>& transpose={0, 1, 2, 3});

	void fill_host_with_2d_vec(const std::vector<std::vector<float>>& array,
			const std::vector<uint32_t>& transpose={0, 1, 2, 3});

    void range(const std::vector<int>& startIndex, const std::vector<int>& endIndex, Data<Dtype>* result);
	void transpose(const std::vector<uint32_t>& t);

	bool compareData(Data<Dtype>* data, const Dtype error = Dtype(0.001), const bool print = true);
	//static bool compareData(Data<Dtype>* data1, Data<Dtype>* data2,
	//		const Dtype error = Dtype(0.001));

	bool compareGrad(Data<Dtype>* data, const Dtype error = Dtype(0.001), const bool print = true);
	//static bool compareGrad(Data<Dtype>* data1,	Data<Dtype>* data2,
	//		const Dtype error = Dtype(0.001));

public:
	//std::shared_ptr<Data<Dtype>> _input;
	std::shared_ptr<SyncMem<Dtype>> _data;				///< Data의 데이터
	std::shared_ptr<SyncMem<Dtype>> _grad;				///< Data의 그레디언트
	std::string _name;

private:
    std::vector<uint32_t> _shape;   ///< Data의 shape, Batches, Channels, Rows, Columns의 
                                    /// 4차원 벡터로 구성

	size_t _count;				    ///< Data 메모리의 크기, 엘레먼트의 수 
                                    /// (Batches*Channels*Rows*Columns)

    // XXX: 현재 hostOnly 플래그의 값과 무관하게 
    // 항상 device memory를 할당하고 있음.
	bool _hostOnly;



public:
	const static uint32_t SHAPE_SIZE = 4;
	static uint32_t printConfig;
};

#endif /* DATA_H_ */
