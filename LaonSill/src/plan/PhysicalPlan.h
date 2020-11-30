/**
 * @file PhysicalPlan.h
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string.h>

#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#include <list>
#include <string>

#include "common.h"
#include "LogicalPlan.h"
#include "LossConsole.h"

#ifndef PHYSICALPLAN_H
#define PHYSICALPLAN_H 

typedef struct PlanInfo_t {
    std::string networkID;
    int         dopCount;
    int         epochCount;
    int         miniBatchCount;     // per epoch

    int         curEpochIndex;
    int         curMiniBatchIndex;
    int         doneCount;       
    std::mutex  planMutex;      // mutex for curEpochIndex, curMiniBatchIndex, doneCount
} PlanInfo;

typedef struct TensorAllocKey_t {
    std::string     tensorName;
    PlanAlloc       tensorAlloc;

    bool operator < (const struct TensorAllocKey_t &x) const {
        if (strcmp(tensorName.c_str(), x.tensorName.c_str()) == 0) {
            if (tensorAlloc.nodeID == x.tensorAlloc.nodeID) {
                return tensorAlloc.devID < x.tensorAlloc.devID;
            } else {
                return tensorAlloc.nodeID < x.tensorAlloc.nodeID;
            }
        } else {
            return tensorName < x.tensorName;
        }
    }
} TensorAllocKey;

class PhysicalPlan {
    friend class WorkContext;

public: 
    PhysicalPlan(std::vector<std::string> lossNames);
    virtual ~PhysicalPlan();

    std::string                 networkID;
    std::map<int, PlanAlloc>    allocMap;       // 특정 레이어를 어느곳에 배치할 것인가
                                                // key : layer ID, value : PlanAlloc
    std::map<int, PlanDef>      planMap;        // plan ID별 PlanDef 맵
                                                // key : planID, value : PlanDef
    std::map<int, int>          depRefMap;      // 각 plan들의 dependency를 관리한다.
                                                // key : planID, value : remain dependecy
                                                // count
    std::map<int, void*>        instanceMap;    // key : layer ID, value : instance pointer
    std::map<PlanType, int>     planTypeRCMap; // Reference Count per plan type map 

    int                         dopID;          // degree of parallelism ID
    int                         refCount;   // 이 값이 0이 되면 해당 mini batch에 대한 plan은
                                            // 다 수행한 것으로 판단하면 된다.

    bool generatePlan(bool genNextMiniBatch, bool genPlanOnly);    
    // 현 minibatch에 해당하는 작업이 완료되면 그다음 mini batch에 대한 플랜을 생성한다.
    // 만약 모든 batch를 다 돌았을 경우에는 false를 반환한다.

    bool runPlan(int layerID, PlanType planType, bool inference);
    bool runPlan(PlanType planType, bool inference);
    bool runPlan(bool inference);

    std::list<int>      readyQueue; // dependency 문제가 해결이 되어 실행이 될 수 있는
                                    // planID들의 리스트
    std::mutex          planMutex;  // refCount, readyQueue, depRefMap을 보호한다.
                                    // XXX: 락을 효율적으로 사용하도록 추후에 수정하자.

    static void insertPlan(std::string networkID, std::vector<PhysicalPlan*> pMap,
        PlanInfo *pInfoMap);
    static void removePlan(std::string networkID);

    static PhysicalPlan* getCurPhysicalPlan();

    static void allocateTensor(std::string networkID);

    static void setCurPlan(std::string networkID, int dopID, bool acquireLock);
    static void setCurPlanInfo(std::string networkID);

    static void saveNetwork(bool checkCond);
    static void loadNetwork();

    static int getDOPCount(std::string networkID);
    static void markFinish(std::string networkID, int dopID, int planID);   

    void* getTensor(int nodeID, int devID, std::string tensorName);

    void reset();
    static void setCurProgress(std::string networkID, int iterNum);

protected:
    static std::map<std::string, std::vector<PhysicalPlan*>>    planGlobalMap;    
                                                                // key = networkID,
                                                                // value = Physical Plans
    static std::map<std::string, PlanInfo*>  planGlobalInfoMap;
                                                // 하나의 네트워크에 대한 plan 정보를 
                                                // 담고 있는 구조체. 
                                                // key = networkID, value = plan info 
    static std::mutex               planGlobalMutex;    // planMap, planInfoMap을 보호

private:
    void runLayer(int planID, bool inference);
    void notifyFinish(int targetPlanID);
    void markDone(int planID);     

    std::vector<int> getOrderedLayerIDs(std::string networkID, bool buildTrainActivation);
    void allocateTensorInternal(std::string networkID, int dopID, bool buildTrainActivation);
    static void* allocTensorMem(int layerType, void* instancePtr, std::string tensorName,
                                PlanAlloc planAlloc, bool isInput, int index);

    LossConsole *lossConsole;       // FIXME: 이름이... 이상하다 ㅠ_ㅠ 좋은걸로 바꿔줘요
    float calcLoss();
    void logLoss();

    std::map<TensorAllocKey, void*> tensorAllocMap;
};
#endif /* PHYSICALPLAN_H */
