/**
 * @file	MockDataSet.h
 * @date	2016/4/23
 * @author	jhkim
 * @brief
 * @details
 */




#ifndef DATASET_MOCKDATASET_H_
#define DATASET_MOCKDATASET_H_

#include "common.h"
#include "DataSet.h"

/**
 * @brief 랜덤 데이터셋 생성을 위해 구현된 DataSet 클래스.
 * @details 생성자에 지정된 파라미터값에 따라 해당 차원의 랜덤 데이터셋을 생성한다.
 *          디버깅 및 테스트용.
 * @todo -0.1 ~ 0.1의 uniform dist의 난수 생성으로 지정, 필요에 따라 변경할 수 있도록 수정.
 */
template <typename Dtype>
class MockDataSet : public DataSet<Dtype> {
public:
	const static uint32_t FULL_RANDOM = 0;
	const static uint32_t NOTABLE_IMAGE = 1;

	MockDataSet(uint32_t rows, uint32_t cols, uint32_t channels, uint32_t numTrainData,
                uint32_t numTestData, uint32_t numLabels, uint32_t mode = 0);
	virtual ~MockDataSet();

	virtual void load();
private:
	void _fillFullRandom();
	void _fillNotableImage();


	uint32_t numLabels;	 ///< 생성할 데이터셋의 정답레이블 크기. 랜덤 레이블 생성을 위해 필요
	uint32_t mode;
};

#endif /* DATASET_MOCKDATASET_H_ */
