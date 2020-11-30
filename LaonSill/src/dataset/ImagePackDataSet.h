/**
 * @file	ImagePackDataSet.h
 * @date	2016/7/13
 * @author	jhkim
 * @brief
 * @details
 */





#ifndef IMAGEPACKDATASET_H_
#define IMAGEPACKDATASET_H_

#include <vector>
#include <string>

#include "common.h"
#include "DataSet.h"


/**
 * @brief 수정된 Mnist 파일 형태의 데이터셋 파일을 읽기위해 구현된 DataSet
 * @details Mnist(http://yann.lecun.com/exdb/mnist/)의 파일 포맷에서
 *          - # of rows, # of columns 후, # of channels 추가.
 *          - 256개 이상의 레이블 사이즈를 수용하기 위해 label 데이터를 
 *            unsigned char -> unsigned int로 변경
 *          (mnist의 경우 10진법 숫자를 구별하기 위해 10개의 카테고리가 있었고 이는 
 *           2^8=256, 8bit으로 수용가능해 unsigned char를 사용)
 * @todo mnist 파일의 수정없이 파라미터를 통해 mnist 원본을 그대로 읽을 수 있게 수정 가능.
 */
template <typename Dtype>
class ImagePackDataSet : public DataSet<Dtype> {
public:
	/**
	 * @details ImagePackDataSet 생성자.
	 * @param trainImage 학습데이터셋 파일 경로.
	 *        파일명은 train_data[숫자] 형식으로 숫자는 0에서 numTrainFile-1의 범위.
	 * @param trainLabel 학습데이터셋 정답레이블 파일 경로.
	 *        파일명은 train_label[숫자] 형식으로 숫자는 0에서 numTrainFile-1의 범위.
	 * @param numTrainFile 학습데이터셋 파일의 수.
	 * @param testImage 테스트데이터셋 파일 경로.
	 *        파일명은 test_data[숫자] 형식으로 숫자는 0에서 numTestFile-1위 범위.
	 * @param testLabel 테스트데이터셋 정답레이블 파일 경로.
	 *        파일명은 test_label[숫자] 형식으로 숫자는 0에서 numTestFile-1위 범위.
	 * @param numTestFile 테스트데이터셋 파일의 수.
	 */
	ImagePackDataSet(
			std::string trainImage,
			std::string trainLabel,
			uint32_t numTrainFile,
			std::string testImage,
			std::string testLabel,
			uint32_t numTestFile);
	virtual ~ImagePackDataSet();

	virtual void load();

	virtual const Dtype* getTrainDataAt(int index);
	virtual const Dtype* getTrainLabelAt(int index);
	virtual const Dtype* getValidationDataAt(int index);
	virtual const Dtype* getValidationLabelAt(int index);
	virtual const Dtype* getTestDataAt(int index);
	virtual const Dtype* getTestLabelAt(int index);



	int load(typename DataSet<Dtype>::Type type, int page);

	virtual void shuffleTrainDataSet();
	virtual void shuffleValidationDataSet();
	virtual void shuffleTestDataSet();


#ifndef GPU_MODE
protected:
	int loadDataSetFromResource(std::string resources[2], DataSample *&dataSet);
#else
protected:

	int loadDataSetFromResource(
			std::string data_path,
			std::string label_path,
			std::vector<Dtype> *&dataSet,
			std::vector<Dtype> *&labelSet,
			std::vector<uint32_t>*& setIndices);

	//virtual void zeroMean(bool hasMean=false, bool isTrain=true);


#endif

protected:
	const std::string trainImage;				///< 학습데이터셋 파일 경로
	const std::string trainLabel;				///< 학습데이터셋 정답레이블 파일 경로
	const int numTrainFile;					///< 학습데이터셋 파일 수
	const std::string testImage;					///< 테스트데이터셋 파일 경로
	const std::string testLabel;					///< 테스트데이터셋 정답레이블 파일 경로
	const int numTestFile;					///< 테스트데이터셋 파일 수
	//const double validationSetRatio;		///< 학습데이터셋의 유효데이터 비율

	int trainFileIndex;						///< 현재 학습데이터 파일 인덱스
	int testFileIndex;						///< 현재 테스트 파일 인덱스
	int numImagesInTrainFile;				///< 학습데이터셋 파일 하나에 들어있는 데이터의 수
	int numImagesInTestFile;			///< 테스트데이터셋 파일 하나에 들어있는 데이터의 수

    std::vector<uint8_t>* bufDataSet;	///< 데이터셋 데이터를 로드할 버퍼. 파일의 uint8_t타입
                                        ///  데이터를 버퍼에 올려 uint32_t타입으로 변환하기 
                                        //   위한 버퍼.
    std::vector<uint32_t>* bufLabelSet;

	/////////////////////////////////////////////////////////////////////////////////////////

	struct thread_arg_t {
		//typename DataSet<Dtype>::Type type;
		void* context;
		int page;
	};

    std::vector<Dtype>* frontTrainDataSet;
    std::vector<Dtype>* frontTrainLabelSet;
    std::vector<Dtype>* backTrainDataSet;
    std::vector<Dtype>* backTrainLabelSet;

    std::vector<uint32_t>* trainFileIndices;

	bool loading;

	pthread_t bufThread;
	thread_arg_t threadArg;
    std::vector<Dtype>* secondTrainDataSet;		///< 학습데이터셋 벡터에 대한 포인터.
    std::vector<Dtype>* secondTrainLabelSet;	///< 학습데이터셋의 정답 레이블 벡터에 대한 
                                                ///  포인터.

    //Dtype scale;

	static void* load_helper(void* context);
	void swap();
};

#endif /* IMAGEPACKDATASET_H_ */
