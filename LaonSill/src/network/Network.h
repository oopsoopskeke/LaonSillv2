/**
 * @file Network.h
 * @date 2016/4/20
 * @author jhkim
 * @brief
 * @details
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include <string>
#include <queue>

#include "common.h"
#include "BaseLayer.h"
#include "InputLayer.h"
#include "LayerConfig.h"
#include "Worker.h"
#include "EnumDef.h"
#include "LogicalPlan.h"

template <typename Dtype> class DataSet;

/**
 * @brief 네트워크 기본 클래스
 * @details 실제 작업이 일어나는 링크드 형식의 레이어 목록을 래핑하고 사용자 인터페이스를 
 * 제공한다.
 * @todo sgd 형태의 네트워크를 기본으로 하고 있으나 다양한 형태의 네트워크 최적화 기법을 
 * 적용할 수 있도록 수정되어야 한다.
 *       (SGD, AdaDelta, AdaGrad, Adam, NAG ... )
 *       개별 파라미터를 각각 전달하는 형태이나 Network Param 형태의 구조체를 만들어 한 번에 
 *       전달하도록 수정해야 한다.
 */
template <typename Dtype>
class Network {
public:
	/**
	 * @details Network 생성자
	 */
	Network();

	/**
	 * @details Network 소멸자
	 */
	virtual ~Network();

	/**
	 * @details Network 초기화 함수
	 */
    static void init();

	/**
	 * @details run()을 수행하면서 시간을 측정한다.
	 * @param epochs run()을 수행할 최대 epoch
	 */
	void run_with_timer(bool inference);

	/**
	 * @details 네트워크를 실행한다.
	 * @param inference     inference 여부를 결정
	 */
	void run(bool inference);

    /**
     * @details 네트워크를 종료한다.
     * @param networkID     종료한 networkID
     */
    static void stopNetwork(std::string networkID);

    /**
     * @details 학습네트워크의 학습을 재개한다.
     * @param networkID     학습을 재개한 networkID
     * @param paramFileName 학습파라미터의 파일 이름. 만약 빈 문자열이면 마지막으로 저장된
     *                      파라미터를 사용한다.
     * @param keepHistory   예전 기록(ex. 예전의 measure, 예전의 train 정보들)의 유지 여부.
     * @return 학습재개를 실패하면 빈 문자열을 반환한다.
     *         성공적으로 동작하는 경우에 학습 재개된 네트워크 ID를 반환한다.
     */
    static std::string createResumeNetwork(std::string networkID, std::string paramFilePath,
            bool keepHistory);

    /**
     * @details 학습중인 네트워크에 대해서 inference를 수행한다.
     * @param inputLayerName input layer name
     * @param channel inference할 이미지의 체널
     * @param height inference할 이미지의 height
     * @param width inference할 이미지의 width
     * @param imageData inference할 이미지의 픽셀 정보
     */
    void runAdhoc(std::string inputLayerName, int channel, int height, int width,
            float* imageData);

	/**
	 * @details 네트워크를 plantype별로 1번의 mini batch를 실행한다. 이 함수를 호출한 이후에
     *          다시 reset()함수를 호출할 필요는 없다.
     * @param planType      planType (forward, backward, update)
	 * @param inference     inference 여부를 결정
	 */
    void runPlanType(PlanType planType, bool inference);

	/**
	 * @details 네트워크를 준비한다.
	 * @param epochs run()을 수행할 최대 epoch
	 */
    void build(int epochs);

	/**
	 * @details 네트워크를 초기화 한다. 한번 네트워크를 실행하고, 다시 그 네트워크를
     *          실행하고자 할때 이 함수를 호출한다.
	 */
    void reset();

	/**
	 * @details minibatch 1회를 수행한다.
     * @param inference     inference 여부를 결정
     * @param miniBatchIdx  수행할 mini batch index
	 */
    void runMiniBatch(bool inference, int miniBatchIdx);

	/**
	 * @details 네트워크를 파일에 쓴다.
	 * @param path 네트워크를 쓸 파일의 경로
	 */
    void save(std::string path);

	/**
	 * @details 네트워크를 파일에 쓴다. 네트워크 파일경로는 미리 지정이 되어 있어야 한다.
	 */
    std::string save();

	/**
	 * @details 네트워크를 파일로부터 읽는다.
	 * @param filename 네트워크를 읽을 파일의 경로
	 */
    void load(std::string path);

	/**
	 * @details 네트워크를 파일로부터 읽는다. 네트워크 파일경로는 미리 지정이 되어 있어야
     *          한다.
	 */
	void load();

	/**
	 * @details 네트워크 내부의 레이어를 이름으로 찾는다.
	 * @param name 찾을 레이어의 이름
	 * @return 찾은 레이어에 대한 포인터
	 */
	Layer<Dtype>* findLayer(const std::string layerName);

	/**
	 * @details 네트워크 내부의 레이어를 이름으로 찾는다.
	 * @param name 찾을 레이어의 이름
	 * @return 찾은 레이어에 대한 포인터
	 */
	Layer<Dtype>* findLayer(const std::string layerName, LayerActivation activation);

	/**
	 * @details 네트워크에서 특정 레이어타입을 가지고 있는 모든 레이어를 반환한다.
	 * @param layerType 찾을 레이어타입
	 * @return 레이어포인트 어레이
	 */
    std::vector<Layer<Dtype>*> findLayersByType(int layerType);

	/**
	 * @details 네트워크에서 특정 텐서를 찾아서 반환한다.
	 * @param nodeID        노드 아이디
     * @param devID         디바이스 아이디
     * @param tensorName    찾을 텐서이름
	 * @return 텐서포인터를 반환
	 */
    Data<Dtype>* findTensor(int nodeID, int devID, std::string tensorName);


    /**
     * @details 네트워크 아이디를 반환한다.
     * @return  네트워크 아이디
     */
    std::string                             getNetworkID() { return this->networkID; }

    /**
     * @details 특정 네트워크 아이디를 가지고 있는 네트워크를 반환한다.
     * @param networkID     네트워크 아이디
     * @return  네트워크 포인터를 반환
     */
    static Network<Dtype>*                  getNetworkFromID(std::string networkID);

    bool                                    isInnerLayer(int layerID);


    /**
     * @details 네트워크 정의가 로드되었음을 설정한다.
     */
    void                                    setLoaded() { this->isLoaded = true; }
    bool                                    getLoaded() { return this->isLoaded; }
    void                                    setStop()   { this->isNeedStop = true; }
    bool                                    getStop()   { return this->isNeedStop; }

    void                                    setBuilt() { this->isBuilt = true; }
    void                                    unsetBuilt() { this->isBuilt = false; }
    bool                                    getBuilt() { return this->isBuilt; }
    bool                                    getMeasureInserted() { 
                                                return this->isMeasureInserted; }
    void                                    setMeasureInserted() {
                                                this->isMeasureInserted = true; }

    static bool                             addAdhocRun(std::string networkID);
    static void                             removeAdhocRun(std::string networkID);

    /*
     * 파라미터 관리를 위한 함수들.
     * FIXME: 추후에 다른 모듈로 분리하자
     */
    void                                    handleIntervalSaveParams(int iterNum);
    void                                    handleBestLoss(float loss, int iterNum); 

    /*
     * 트레인 파일 관리를 위한 함수들.
     * FIXME: 추후에 다른 모듈로 분리하자
     */
    void                                    logNetworkDefString(std::string networkDef);
    void                                    logNetworkDefFile(std::string networkDefFilePath);
    void                                    logTrainFile(std::string content);
    void                logTrainHistory(std::vector<std::pair<int, std::string>> params);
    void        logMeasureHistory(std::vector<std::pair<int, std::vector<float>>> measures);

    static bool                     getTrainInfo(std::string networkID,
                                        std::string &networkDef,
                                        std::vector<std::pair<int, std::string>> &params);
    static bool             getMeasureInfo(std::string networkID, int targetIter,
                                std::vector<std::pair<int, std::vector<float>>> &measures);
private:
    std::string                                     networkID;
    static std::map<std::string, Network<Dtype>*>   networkIDMap;
    static std::mutex                               networkIDMapMutex;
    bool                                            isLoaded;
    bool                                            isBuilt;
    bool                                            isMeasureInserted;
    volatile bool                                   isNeedStop;

    volatile int                                    adhocRunRefCount;
    std::mutex                                      adhocRunMutex;

    /*
     * 파라미터 관리를 위한 변수들.
     * FIXME: 추후에 다른 모듈로 분리하자
     */
    float                                           bestLoss;
    std::string                                     bestSavedParamPath;
    std::queue<std::string>                         intervalSavedParamPathQueue;

    /*
     * 트레인 파일 관리를 위한 변수들.
     * FIXME: 추후에 다른 모듈로 분리하자
     */
    FILE                                           *trainFP;
};


#endif /* NETWORK_H_ */
