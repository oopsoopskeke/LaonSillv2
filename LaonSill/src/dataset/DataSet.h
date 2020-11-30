/*
 * DataSet.h
 *
 *  Created on: 2016. 8. 16.
 *      Author: jhkim
 */

#ifndef DATASET_H_
#define DATASET_H_


#include <numeric>
#include <vector>
#include <algorithm>

#include "common.h"
#include "Util.h"

template <typename Dtype>
class DataSet {
public:
	DataSet();
	DataSet(uint32_t rows, uint32_t cols, uint32_t channels, uint32_t numTrainData,
        uint32_t numTestData);
	virtual ~DataSet();

	/**
	 * @details 데이터 하나의 행 수를 조회
	 * @return 데이터 하나의 행 수
	 */
	uint32_t getRows() const { return this->rows; }
	/**
	 * @details 데이터 하나의 열 수를 조회
	 * @return 데이터 하나의 열 수
	 */
	uint32_t getCols() const { return this->cols; }
	/**
	 * @details 데이터 하나의 채널 수를 조회
	 * @return 데이터 하나의 채널 수
	 */
	uint32_t getChannels() const { return this->channels; }
	/**
	 * @details 학습데이터의 수를 조회
	 * @return 학습 데이터의 수
	 */
	uint32_t getNumTrainData() const { return this->numTrainData; }
	/**
	 * @details 유효데이터의 수를 조회
	 * @return 유효데이터의 수
	 */
	uint32_t getNumValidationData() const { return this->numValidationData; }
	/**
	 * @details 테스트데이터의 수를 조회
	 * @return 테스트데이터의 수
	 */
	uint32_t getNumTestData() const { return this->numTestData; }
	/**
	 * @details 학습데이터에 대한 포인터 조회.
	 * @return 학습데이터에 대한 포인터.
	 */
	const std::vector<Dtype> *getTrainDataSet() const { return this->trainDataSet; }
	/**
	 * @details 유효데이터에 대한 포인터 조회.
	 * @return 유효데이터에 대한 포인터.
	 */
	const std::vector<Dtype> *getValidationDataSet() const { return this->validationDataSet; }
	/**
	 * @details 테스트데이터에 대한 포인터 조회.
	 * @return 테스트데이터에 대한 포인터.
	 */
	const std::vector<Dtype> *getTestDataSet() const { return this->testDataSet; }
	/**
	 * @details 특정 채널의 평균값을 조회한다.
	 * @param channel 조회하고자 하는 채널의 index
	 * @return 지정한 채널의 평균값.
	 */
	//const Dtype getMean(uint32_t channel) const { return mean[channel]; }
	/**
	 * @details 전체 채널의 평균값 배열의 첫번째 위치에 대한 포인터를 조회한다.
	 * @return 전체 채널 평균값 배열의 첫번째 위치에 대한 포인터.
	 */
	//const Dtype *getMean() const { return mean; }

	//void setMean(const std::vector<Dtype>& means);

	/**
	 * @details index번째 학습데이터에 대한 포인터 조회.
	 * @param index 조회하고자 하는 학습 데이터의 index
	 * @return index번째 학습데이터에 대한 포인터.
	 */
	virtual const Dtype* getTrainDataAt(int index);
	/**
	 * @details index번째 학습데이터의 정답 레이블에 대한 포인터 조회.
	 * @param index 조회하고자 하는 학습 데이터 정답 레이블의 index
	 * @return index번째 학습데이터 정답 레이블에 대한 포인터.
	 */
	virtual const Dtype* getTrainLabelAt(int index);
	/**
	 * @details index번째 유효데이터에 대한 포인터 조회.
	 * @param index 조회하고자 하는 유효 데이터의 index
	 * @return index번째 유효데이터에 대한 포인터.
	 */
	virtual const Dtype* getValidationDataAt(int index);
	/**
	 * @details index번째 유효데이터의 정답 레이블에 대한 포인터 조회.
	 * @param index 조회하고자 하는 유효 데이터 정답 레이블의 index
	 * @return index번째 유효데이터 정답 레이블에 대한 포인터.
	 */
	virtual const Dtype* getValidationLabelAt(int index);
	/**
	 * @details index번째 테스트데이터에 대한 포인터 조회.
	 * @param index 조회하고자 하는 테스트데이터의 index
	 * @return index번째 테스트데이터에 대한 포인터.
	 */
	virtual const Dtype* getTestDataAt(int index);
	/**
	 * @details index번째 테스트데이터의 정답 레이블에 대한 포인터 조회.
	 * @param index 조회하고자 하는 테스트데이터 정답 레이블의 index
	 * @return index번째 테스트데이터 정답 레이블에 대한 포인터.
	 */
	virtual const Dtype* getTestLabelAt(int index);

	/**
	 * @details 학습데이터의 각 채널별 평균을 구하고 학습, 유효, 테스트 데이터에 대해 
     *          평균만큼 shift.
	 * @param hasMean 이미 계산된 평균값이 있는지 여부, 미리 계산된 평균값이 있는 경우 다시
     *                평균을 계산하지 않는다.
	 */
	//virtual void zeroMean(bool hasMean=false);

	/**
	 * @details 학습,유효,테스트 데이터를 메모리로 로드.
	 */
	virtual void load() = 0;
	/**
	 * @details 학습데이터를 임의의 순서로 섞는다.
	 */
	virtual void shuffleTrainDataSet();
	/**
	 * @details 유효데이터를 임의의 순서로 섞는다.
	 */
	virtual void shuffleValidationDataSet();
	/**
	 * @details 테스트데이터를 임의의 순서로 섞는다.
	 */
	virtual void shuffleTestDataSet();

protected:
	enum Type {
		Train = 0,		//학습 데이터
		Validation = 1,	//검증 데이터
		Test = 2		//테스트 데이터
	};

	uint32_t rows;					///< 데이터의 rows (height)값.
	uint32_t cols;					///< 데이터의 cols (width)값.
	uint32_t channels;				///< 데이터의 channel값.
	size_t dataSize;				///< 데이터셋 데이터 하나의 요소수 (rows*cols*channels)

	uint32_t numTrainData;						///< 전체 학습데이터의 수.
	uint32_t numValidationData;					///< 전체 유효데이터의 수.
	uint32_t numTestData;						///< 전체 테스트데이터의 수.

    std::vector<Dtype>* trainDataSet;		///< 학습데이터셋 벡터에 대한 포인터.
    std::vector<Dtype>* trainLabelSet;		///< 학습데이터셋의 정답 레이블 벡터에 대한 포인터.
    std::vector<uint32_t>* trainSetIndices;	///< 학습셋 인덱스 벡터 포인터. 데이터와 레이블을
                                            ///  함께 shuffle하기 위한 별도의 인덱스 벡터.

    std::vector<Dtype>* validationDataSet;	///< 유효데이터셋 벡터에 대한 포인터.
    std::vector<Dtype>* validationLabelSet;	///< 유효데이터셋의 정답 레이블 벡터에 대한 포인터
    std::vector<uint32_t>* validationSetIndices;	///< 유효셋 인덱스 벡터 포인터 데이터와 
                                                    ///  레이블을 함께 shuffle하기 위한 별도의
                                                    ///  인덱스 벡터.

    std::vector<Dtype>* testDataSet;	///< 테스트데이터셋 벡터에 대한 포인터.
    std::vector<Dtype>* testLabelSet;	///< 테스트데이터셋의 정답 레이블 벡터에 대한 포인터.
    std::vector<uint32_t>* testSetIndices;	///< 테스트셋 인덱스 벡터 포인터. 데이터와 
                                            ///  레이블을 함께 shuffle하기 위한 별도의 인덱스
                                            ///  벡터.

	//Dtype mean[3];				///< 학습데이터셋의 각 채널별 평균값을 저장하는 배열.
};

#endif /* DATASET_H_ */
