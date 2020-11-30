/*
 * Data.cpp
 *
 *  Created on: 2016. 8. 19.
 *      Author: jhkim
 */

#include <string.h>
#include <cstdlib>
#include <string>

#include "Cuda.h"
#include "Data.h"
#include "SysLog.h"
#include "MemoryMgmt.h"

//#define DATA_LOG
using namespace std;


template <typename Dtype>
uint32_t Data<Dtype>::printConfig = 0;


template <typename Dtype>
Data<Dtype>::Data(const string& name, const bool hostOnly)
: _name(name), _count(0), _hostOnly(hostOnly) {
	this->_shape.resize(SHAPE_SIZE);

    this->_data = std::make_shared<SyncMem<Dtype>>();
    this->_grad = std::make_shared<SyncMem<Dtype>>();
}

template <typename Dtype>
Data<Dtype>::Data(Data<Dtype>* data, const bool hostOnly)
: Data(data->_name, hostOnly) {
	this->reshape(data->getShape());
	this->set_host_data(data->host_data());
	this->set_host_grad(data->host_grad());
}

template <typename Dtype>
Data<Dtype>::Data(const string& name, Data<Dtype>* data, uint32_t type, const bool hostOnly)
: _name(name), _count(0), _hostOnly(hostOnly) {
	this->_shape.resize(SHAPE_SIZE);

	// type 0: data share, grad 별도
	if (type == 0) {
		this->_data = data->_data;
        this->_grad = std::make_shared<SyncMem<Dtype>>();
	}
	// type 1: data 별도, grad share
	else if(type == 1) {
        this->_data = std::make_shared<SyncMem<Dtype>>();
		this->_grad = data->_grad;
	}
}

template <typename Dtype>
Data<Dtype>::Data(const string& name, const vector<uint32_t>& _shape, const bool hostOnly)
: Data(name, hostOnly) {
	reshape(_shape);
}

template <typename Dtype>
void Data<Dtype>::share_data(Data<Dtype>* data) {
	this->_data = data->_data;
}

template <typename Dtype>
void Data<Dtype>::share_grad(Data<Dtype>* data) {
	this->_grad = data->_grad;
}

template <typename Dtype>
Data<Dtype>::~Data() {}

bool isValidShapeSize(const vector<uint32_t>& shape, const uint32_t shapeSize) {
	return shape.size() == shapeSize;
}

bool isValidShapeValue(const vector<uint32_t>& shape) {
	for (uint32_t i = 0; i < shape.size(); i++) {
		if(shape[i] < 0) {
			return false;
		}
	}
	return true;
}

template <typename Dtype>
void Data<Dtype>::reshape(const vector<uint32_t>& shape) {
	SASSERT0(isValidShapeSize(shape, SHAPE_SIZE));
	SASSERT0(isValidShapeValue(shape));
	_shape = shape;

	_count = 1;
	for(uint32_t i = 0; i < _shape.size(); i++) {
        _count *= _shape[i];
    }

	_data->reshape(_count);
    _grad->reshape(_count);
}

template <typename Dtype>
void Data<Dtype>::reshapeInfer(const vector<int>& shape) {
	vector<uint32_t> fShape(shape.size());
	int inferredAxis = -1;

	uint32_t newFixedCount = 1;
	for (uint32_t i = 0; i < shape.size(); i++) {
		if (shape[i] < 0) {
			SASSERT0(inferredAxis == -1);
			inferredAxis = i;
		} else if(shape[i] == 0) {
			fShape[i] = this->_shape[i];
			newFixedCount *= this->_shape[i];
		} else {
			fShape[i] = shape[i];
			newFixedCount *= shape[i];
		}
	}

	SASSERT0(_count % newFixedCount == 0);
	fShape[inferredAxis] = _count / newFixedCount;

	reshape(fShape);
}

template <typename Dtype>
void Data<Dtype>::reshapeLike(const Data<Dtype>* data) {
	reshape(data->getShape());
}

template <typename Dtype>
const Dtype* Data<Dtype>::host_data() {
	return _data->host_mem();
}

template <typename Dtype>
const Dtype* Data<Dtype>::device_data() {
	SASSERT0(!this->_hostOnly);
	return _data->device_mem();
}

template <typename Dtype>
const Dtype* Data<Dtype>::host_grad() {
	return _grad->host_mem();
}

template <typename Dtype>
const Dtype* Data<Dtype>::device_grad() {
	SASSERT0(!this->_hostOnly);
	return _grad->device_mem();
}

template <typename Dtype>
Dtype* Data<Dtype>::mutable_host_data() {
	return _data->mutable_host_mem();
}

template <typename Dtype>
Dtype* Data<Dtype>::mutable_device_data() {
	SASSERT0(!this->_hostOnly);
	return _data->mutable_device_mem();
}

template <typename Dtype>
Dtype* Data<Dtype>::mutable_host_grad() {
	return _grad->mutable_host_mem();
}

template <typename Dtype>
Dtype* Data<Dtype>::mutable_device_grad() {
	SASSERT0(!this->_hostOnly);
	return _grad->mutable_device_mem();
}

template <typename Dtype>
void Data<Dtype>::reset_host_data(const bool setZero, const Dtype value) {
	_data->reset_host_mem(setZero, value);
}

template <typename Dtype>
void Data<Dtype>::reset_device_data(const bool setZero, const Dtype value) {
	SASSERT0(!this->_hostOnly);
	_data->reset_device_mem(setZero, value);
}

template <typename Dtype>
void Data<Dtype>::reset_host_grad() {
	_grad->reset_host_mem();
}

template <typename Dtype>
void Data<Dtype>::reset_device_grad() {
	SASSERT0(!this->_hostOnly);
	_grad->reset_device_mem();
}



template <typename Dtype>
void Data<Dtype>::set(Data<Dtype>* data, bool reshape, int type) {
	if (reshape)
		this->reshapeLike(data);

	SASSERT0(_count == data->getCount());

	if (this->_hostOnly || data->_hostOnly) {
		if (type == 0 || type == 1)
			this->set_host_data(data);
		if (type == 0 || type == 2)
			this->set_host_grad(data);
	} else {
		if (type == 0 || type == 1)
			this->set_device_data(data);
		if (type == 0 || type == 2)
			this->set_device_grad(data);
	}
}


template <typename Dtype>
void Data<Dtype>::set_host_data(const Dtype* data, const uint32_t offset) {
	_data->set_mem(data, SyncMemCopyType::HostToHost, offset);
}

template <typename Dtype>
void Data<Dtype>::set_host_with_device_data(const Dtype* data) {
	_data->set_mem(data, SyncMemCopyType::DeviceToHost);
}

template <typename Dtype>
void Data<Dtype>::set_device_with_host_data(const Dtype* data, const size_t offset,
    const size_t size) {
	SASSERT0(!this->_hostOnly);
	_data->set_mem(data, SyncMemCopyType::HostToDevice, offset, size);
}

template <typename Dtype>
void Data<Dtype>::set_device_data(const Dtype* data) {
	SASSERT0(!this->_hostOnly);
	_data->set_mem(data, SyncMemCopyType::DeviceToDevice);
}

template <typename Dtype>
void Data<Dtype>::set_host_grad(const Dtype* grad) {
	_grad->set_mem(grad, SyncMemCopyType::HostToHost);
}

template <typename Dtype>
void Data<Dtype>::set_device_grad(const Dtype* grad) {
	SASSERT0(!this->_hostOnly);
	_grad->set_mem(grad, SyncMemCopyType::DeviceToDevice);
}

template <typename Dtype>
void Data<Dtype>::add_host_data(const Dtype* data) {
	_data->add_host_mem(data);
}

template <typename Dtype>
void Data<Dtype>::add_device_data(const Dtype* data) {
	SASSERT0(!this->_hostOnly);
	_data->add_device_mem(data);
}

template <typename Dtype>
void Data<Dtype>::add_host_grad(const Dtype* grad) {
	_grad->add_host_mem(grad);
}

template <typename Dtype>
void Data<Dtype>::add_device_grad(const Dtype* grad) {
	SASSERT0(!this->_hostOnly);
	_grad->add_device_mem(grad);
}

template <typename Dtype>
void Data<Dtype>::scale_host_data(const float scale) {
	_data->scale_host_mem(scale);
}

template <typename Dtype>
void Data<Dtype>::scale_device_data(const float scale) {
	SASSERT0(!this->_hostOnly);
	_data->scale_device_mem(scale);
}

template <typename Dtype>
void Data<Dtype>::scale_host_grad(const float scale) {
	_grad->scale_host_mem(scale);
}

template <typename Dtype>
void Data<Dtype>::scale_device_grad(const float scale) {
	SASSERT0(!this->_hostOnly);
	_grad->scale_device_mem(scale);
}

template <typename Dtype>
double Data<Dtype>::sumsq_device_data() {
	SASSERT0(!this->_hostOnly);
	return _data->sumsq_device_mem();
}

template <typename Dtype>
double Data<Dtype>::sumsq_device_grad() {
	SASSERT0(!this->_hostOnly);
	return _grad->sumsq_device_mem();
}

template <typename Dtype>
double Data<Dtype>::asum_device_data() {
	SASSERT0(!this->_hostOnly);
	return _data->asum_device_mem();
}

template <typename Dtype>
double Data<Dtype>::asum_device_grad() {
	SASSERT0(!this->_hostOnly);
	return _grad->asum_device_mem();
}

template <typename Dtype>
void Data<Dtype>::save(const string& filename) {
	ofstream ofs(filename.c_str(), ios::out | ios::binary);

	const uint32_t numParams = 1;
	ofs.write((char*)&numParams, sizeof(uint32_t));
	save(ofs);

	ofs.close();
}

template <typename Dtype>
void Data<Dtype>::save(ofstream& ofs) {
	// _name
	Util::saveStringToFstream(ofs, this->_name);
	// _shape
	for(uint32_t i = 0; i < SHAPE_SIZE; i++) {
		ofs.write((char*)&_shape[i], sizeof(uint32_t));
	}
	// _data
	_data->save(ofs);
	// grad의 경우 load할 필요가 없어 일단 생략한다.
	//_grad->save(ofs);
}

template <typename Dtype>
void Data<Dtype>::load(const string& filename) {
	ifstream ifs(filename.c_str(), ios::in | ios::binary);
    SASSERT0(ifs.is_open());

	uint32_t numParams;
	ifs.read((char*)&numParams, sizeof(uint32_t));
	load(ifs);

	ifs.close();
}

template <typename Dtype>
void Data<Dtype>::load(ifstream& ifs) {

    SASSERT0(ifs.is_open());
	// _name
	Util::loadStringFromFstream(ifs, this->_name);
	// _shape
	const vector<uint32_t> shape(SHAPE_SIZE);
	for(uint32_t i = 0; i < SHAPE_SIZE; i++) {
		ifs.read((char*)&shape[i], sizeof(uint32_t));
	}
	reshape(shape);
	// _data
	_data->load(ifs);
	//_grad->load(ifs);
}

template <typename Dtype>
void Data<Dtype>::print() {
	if (!printConfig) return;

	cout << this->_name << ": " << _shape[0] << " x " << _shape[1] << " x " <<
			_shape[2] << " x " << _shape[3] << endl;
}

template <typename Dtype>
void Data<Dtype>::print_data(const string& head, const vector<uint32_t>& shape,
		const bool cmo, const int summary) {
	if (!printConfig) return;

	if (shape.size() > 0)
		this->_data->print(head, shape, cmo, true, summary);
	else
		this->_data->print(head, this->_shape, cmo, true, summary);
}

template <typename Dtype>
void Data<Dtype>::print_data(const vector<uint32_t>& shape, const bool cmo,
		const int summary) {
	print_data(_name+"-data", shape, cmo, summary);
}

template <typename Dtype>
void Data<Dtype>::print_data_flatten() {
	if (!printConfig) return;

	_data->print(_name+"-data");
}

template <typename Dtype>
void Data<Dtype>::print_grad(const string& head, const vector<uint32_t>& shape,
		const bool cmo, const int summary) {
	if (!printConfig) return;

	if (shape.size() > 0)
		_grad->print(head, shape, cmo, true, summary);
	else
		_grad->print(head, _shape, cmo, true, summary);
}

template <typename Dtype>
void Data<Dtype>::print_grad(const vector<uint32_t>& shape, const bool cmo,
		const int summary) {
	print_grad(_name+"-grad", shape, cmo, summary);
}

template <typename Dtype>
void Data<Dtype>::print_shape() {
	cout << this->_name << "\'s shape: (";
	for (int i = 0; i < this->_shape.size(); i++) {
		cout << this->_shape[i] << ", ";
	}
	cout << ")" << endl;
}

template <typename Dtype>
void Data<Dtype>::fill_host_with_1d_vec(const vector<int>& array,
			const vector<uint32_t>& transpose) {
	SASSERT0(array.size() > 0);
	const uint32_t dim1 = array.size();

	Dtype* dataPtr = mutable_host_data();
	const uint32_t batchSize = _shape[1]*_shape[2]*_shape[3];
	const uint32_t heightSize = _shape[2]*_shape[3];
	const uint32_t widthSize = _shape[3];

	const uint32_t tBatchSize = 
        _shape[transpose[1]] * _shape[transpose[2]] * _shape[transpose[3]];
	const uint32_t tHeightSize = _shape[transpose[2]]*_shape[transpose[3]];
	const uint32_t tWidthSize = _shape[transpose[3]];

	uint32_t s[4];
	uint32_t& ts0 = s[transpose[0]];
	uint32_t& ts1 = s[transpose[1]];
	uint32_t& ts2 = s[transpose[2]];
	uint32_t& ts3 = s[transpose[3]];

	// batch
	for (s[0] = 0; s[0] < _shape[0]; s[0]++) {
		// height
		for (s[1] = 0; s[1] < _shape[1]; s[1]++) {
			// width
			for (s[2] = 0; s[2] < _shape[2]; s[2]++) {
				// Anchors
				for (s[3] = 0; s[3] < _shape[3]; s[3]++) {
					dataPtr[ts0*tBatchSize+ts1*tHeightSize+ts2*tWidthSize+ts3]
							= int(array[s[0]*batchSize+s[1]*heightSize+s[2]*widthSize+s[3]]);
				}
			}
		}
	}
}

template <typename Dtype>
void Data<Dtype>::fill_host_with_1d_vec(const vector<uint32_t>& array,
			const vector<uint32_t>& transpose) {
	SASSERT0(array.size() > 0);
	const uint32_t dim1 = array.size();

	Dtype* dataPtr = mutable_host_data();
	const uint32_t batchSize = _shape[1]*_shape[2]*_shape[3];
	const uint32_t heightSize = _shape[2]*_shape[3];
	const uint32_t widthSize = _shape[3];

	const uint32_t tBatchSize = _shape[transpose[1]]*_shape[transpose[2]]*_shape[transpose[3]];
	const uint32_t tHeightSize = _shape[transpose[2]]*_shape[transpose[3]];
	const uint32_t tWidthSize = _shape[transpose[3]];

	uint32_t s[4];
	uint32_t& ts0 = s[transpose[0]];
	uint32_t& ts1 = s[transpose[1]];
	uint32_t& ts2 = s[transpose[2]];
	uint32_t& ts3 = s[transpose[3]];

	// batch
	for (s[0] = 0; s[0] < _shape[0]; s[0]++) {
		// height
		for (s[1] = 0; s[1] < _shape[1]; s[1]++) {
			// width
			for (s[2] = 0; s[2] < _shape[2]; s[2]++) {
				// Anchors
				for (s[3] = 0; s[3] < _shape[3]; s[3]++) {
					dataPtr[ts0*tBatchSize+ts1*tHeightSize+ts2*tWidthSize+ts3]
			            = uint32_t(array[s[0]*batchSize+s[1]*heightSize+s[2]*widthSize+s[3]]);
				}
			}
		}
	}
}

template <typename Dtype>
void Data<Dtype>::fill_host_with_2d_vec(const vector<vector<float>>& array,
		const vector<uint32_t>& transpose) {
	//SASSERT0(array.size() > 0);
	if (array.size() == 0) {
		return;
	}


	const uint32_t dim1 = array.size();
	const uint32_t dim2 = array[0].size();

	SASSERT0(_shape[3]%dim2 == 0);

	const uint32_t tBatchSize = 
        _shape[transpose[1]]*_shape[transpose[2]]*_shape[transpose[3]];
	const uint32_t tHeightSize = _shape[transpose[2]]*_shape[transpose[3]];
	const uint32_t tWidthSize = _shape[transpose[3]];

	Dtype* dataPtr = mutable_host_data();
	const uint32_t shape3 = _shape[3] / dim2;
	const uint32_t batchSize = _shape[1]*_shape[2]*shape3;
	const uint32_t heightSize = _shape[2]*shape3;
	const uint32_t widthSize = shape3;

	uint32_t s[4];
	uint32_t& ts0 = s[transpose[0]];
	uint32_t& ts1 = s[transpose[1]];
	uint32_t& ts2 = s[transpose[2]];
	uint32_t& ts3 = s[transpose[3]];

	uint32_t q, r;
	// batch
	for (s[0] = 0; s[0] < _shape[0]; s[0]++) {
		// height
		for (s[1] = 0; s[1] < _shape[1]; s[1]++) {
			// width
			for (s[2] = 0; s[2] < _shape[2]; s[2]++) {
				// Anchors
				for (s[3] = 0; s[3] < _shape[3]; s[3]++) {
					q = s[3] / dim2;
					r = s[3] % dim2;
					dataPtr[ts0*tBatchSize+ts1*tHeightSize+ts2*tWidthSize+ts3] =
							array[s[0]*batchSize+s[1]*heightSize+s[2]*widthSize+q][r];
				}
			}
		}
	}
}

template <typename Dtype>
void Data<Dtype>::range(const vector<int>& startIndex,
		const vector<int>& endIndex, Data<Dtype>* result) {

	const uint32_t shapeSize = _shape.size();
	SASSERT0(startIndex.size() == shapeSize);
	SASSERT0(endIndex.size() == shapeSize);

	vector<uint32_t> fStartIndex(shapeSize);
	vector<uint32_t> fEndIndex(shapeSize);

	for (uint32_t i = 0; i < shapeSize; i++) {
		if (startIndex[i] < 0)
			fStartIndex[i] = 0;
		else
			fStartIndex[i] = startIndex[i];
		SASSERT0(fStartIndex[i] < _shape[i]);
	}

	for (uint32_t i = 0; i < shapeSize; i++) {
		if (endIndex[i] < 0)
			fEndIndex[i] = _shape[i];
		else
			fEndIndex[i] = endIndex[i];
		SASSERT0(fEndIndex[i] <= _shape[i]);
		SASSERT0(fEndIndex[i] > fStartIndex[i]);
	}

	//Data<Dtype>* result = NULL;
	//SNEW(result, Data<Dtype>, "result");
	SASSUME0(result != NULL);
	result->reshape({fEndIndex[0]-fStartIndex[0], fEndIndex[1]-fStartIndex[1],
		fEndIndex[2]-fStartIndex[2], fEndIndex[3]-fStartIndex[3]});

	const uint32_t s0Size = getCountByAxis(1);
	const uint32_t s1Size = getCountByAxis(2);
	const uint32_t s2Size = getCountByAxis(3);

	const uint32_t d0Size = result->getCountByAxis(1);
	const uint32_t d1Size = result->getCountByAxis(2);
	const uint32_t d2Size = result->getCountByAxis(3);

	const Dtype* srcData = mutable_host_data();
	Dtype* dstData = result->mutable_host_data();

	for (uint32_t i = fStartIndex[0]; i < fEndIndex[0]; i++) {
		for (uint32_t j = fStartIndex[1]; j < fEndIndex[1]; j++) {
			for (uint32_t k = fStartIndex[2]; k < fEndIndex[2]; k++) {
				for (uint32_t l = fStartIndex[3]; l < fEndIndex[3]; l++) {
					dstData[(i-fStartIndex[0])*d0Size + (j-fStartIndex[1])*d1Size +
					        (k-fStartIndex[2])*d2Size + (l-fStartIndex[3])] =
							srcData[i*s0Size + j*s1Size + k*s2Size + l];
				}
			}
		}
	}

	//return result;
}


template <typename Dtype>
void Data<Dtype>::transpose(const vector<uint32_t>& t) {

    std::shared_ptr<Data<Dtype>> temp = std::make_shared<Data<Dtype>>("temp");
    //Data<Dtype>* temp = NULL;
	//SNEW(temp, Data<Dtype>, "temp");
	//SASSUME0(temp != NULL);
	temp->reshape({_shape[t[0]], _shape[t[1]], _shape[t[2]], _shape[t[3]]});

	const uint32_t s0Size = this->getCountByAxis(1);
	const uint32_t s1Size = this->getCountByAxis(2);
	const uint32_t s2Size = this->getCountByAxis(3);

	const uint32_t d0Size = temp->getCountByAxis(1);
	const uint32_t d1Size = temp->getCountByAxis(2);
	const uint32_t d2Size = temp->getCountByAxis(3);

	uint32_t sIndex[4];
	uint32_t& d0Index = sIndex[t[0]];
	uint32_t& d1Index = sIndex[t[1]];
	uint32_t& d2Index = sIndex[t[2]];
	uint32_t& d3Index = sIndex[t[3]];

	const Dtype* srcData = host_data();
	Dtype* dstData = temp->mutable_host_data();

	for (sIndex[0] = 0; sIndex[0] < this->_shape[0]; sIndex[0]++) {
		for (sIndex[1] = 0; sIndex[1] < this->_shape[1]; sIndex[1]++) {
			for (sIndex[2] = 0; sIndex[2] < this->_shape[2]; sIndex[2]++) {
				for (sIndex[3] = 0; sIndex[3] < this->_shape[3]; sIndex[3]++) {
					dstData[d0Index*d0Size + d1Index*d1Size + d2Index*d2Size + d3Index] =
						srcData[sIndex[0]*s0Size + sIndex[1]*s1Size + 
                                sIndex[2]*s2Size + sIndex[3]];
				}
			}
		}
	}
	set_host_data(temp.get());
    //SDELETE(temp);

	this->_shape = {d0Index, d1Index, d2Index, d3Index};
}

template <typename Dtype>
bool Data<Dtype>::compareData(
		Data<Dtype>* data,
		const Dtype error, const bool print) {

	//SASSERT0(this->getShape() == data->getShape());
	//SASSERT0(this->getCount() == data->getCount());
	//if (this->getShape() != data->getShape()) {
	if (this->getCount() != data->getCount()) {
		cout << "data count inconsistent! at Data::compareData() ... " << endl;
		this->print_shape();
		data->print_shape();
		exit(1);
	}

	const uint32_t count = this->getCount();
	const Dtype* data1Ptr = this->host_data();
	const Dtype* data2Ptr = data->host_data();

	bool result = true;
	const uint32_t batches = this->batches();
	const uint32_t channels = this->channels();
	const uint32_t height = this->height();
	const uint32_t width = this->width();

	const uint32_t channelSize = this->getCountByAxis(1);
	const uint32_t heightSize = this->getCountByAxis(2);
	const uint32_t widthSize = this->getCountByAxis(3);

	size_t errorCnt = 0;
	for (uint32_t i = 0; i < batches; i++) {
		for (uint32_t j = 0; j < channels; j++) {
			for (uint32_t k = 0; k < height; k++) {
				for (uint32_t l = 0; l < width; l++) {
					const uint32_t index = i*channelSize + j*heightSize + k*widthSize + l;
					/*
					if (data1Ptr[index] * data2Ptr[index] < 0) {
						cout << "---data is opposite at (" << i << "," << j << "," <<
								k << "," << l << ")" << endl;
						cout << "data1 is " << data1Ptr[index] << " and data2 is " <<
								data2Ptr[index] << endl;
					}
					*/
					float diff = fabs(data1Ptr[index]-data2Ptr[index]);
					if (diff > error) {
						if (print && errorCnt < 10) {
							cout << "data is different at (" << i << "," << j << "," <<
									k << "," << l << ")" << endl;
							cout << "SoooAComputed: " << data1Ptr[index] << ", CaffeComputed: " <<
									data2Ptr[index] << ", diff: " << diff << endl;
						}
						errorCnt++;
						result = false;
					}
				}
			}
		}
	}
	cout << "compared data <" << this->_name << ", " <<
			data->_name << ">: " << result << "(" << errorCnt << ")" << endl;
	return result;
}


template <typename Dtype>
bool Data<Dtype>::compareGrad(
		Data<Dtype>* data,
		const Dtype error, const bool print) {

	//SASSERT0(this->getShape() == data->getShape());
	//SASSERT0(this->getCount() == data->getCount());
	//if (this->getShape() != data->getShape()) {
	if (this->getCount() != data->getCount()) {
		cout << "data count inconsistent! at Data::compareGrad() ... " << endl;
		this->print_shape();
		data->print_shape();
		exit(1);
	}

	const uint32_t count = this->getCount();
	const Dtype* data1Ptr = this->host_grad();
	const Dtype* data2Ptr = data->host_grad();

	bool result = true;
	const uint32_t batches = this->batches();
	const uint32_t channels = this->channels();
	const uint32_t height = this->height();
	const uint32_t width = this->width();

	const uint32_t channelSize = this->getCountByAxis(1);
	const uint32_t heightSize = this->getCountByAxis(2);
	const uint32_t widthSize = this->getCountByAxis(3);

	size_t errorCnt = 0;
	for (uint32_t i = 0; i < batches; i++) {
		for (uint32_t j = 0; j < channels; j++) {
			for (uint32_t k = 0; k < height; k++) {
				for (uint32_t l = 0; l < width; l++) {
					const uint32_t index = i*channelSize + j*heightSize + k*widthSize + l;
					float diff = fabs(data1Ptr[index]-data2Ptr[index]);
					if (diff > error) {
						if (print && errorCnt < 10) {
							cout << "grad is different at (" << i << "x" << j << "x" <<
									k << "x" << l << ")" << endl;
							cout << "SoooAComputed: " << data1Ptr[index] << ", CaffeComputed: " <<
									data2Ptr[index] << ", diff: " << diff << endl;
						}
						errorCnt++;
						result = false;
					}
				}
			}
		}
	}
	cout << "compared grad <" << this->_name << ", " <<
			data->_name << ">: " << result << "(" << errorCnt << ")" << endl;
	return result;
}



template class Data<float>;
template class Data<int>;
