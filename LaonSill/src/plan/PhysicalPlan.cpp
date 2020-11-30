/**
 * @file PhysicalPlan.cpp
 * @date 2017-05-10
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "PhysicalPlan.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "WorkContext.h"
#include "Param.h"
#include "Worker.h"
#include "LayerFunc.h"
#include "StdOutLog.h"
#include "LearnableLayer.h"
#include "Donator.h"
#include "Network.h"
#include "Task.h"
#include "ThreadMgmt.h"
#include "SysLog.h"
#include "LossLayer.h"
#include "MeasureManager.h"
#include "MeasureLayer.h"
#include "ColdLog.h"
#include "MemoryMgmt.h"

using namespace std;

map<std::string, vector<PhysicalPlan*>> PhysicalPlan::planGlobalMap;
map<std::string, PlanInfo*>             PhysicalPlan::planGlobalInfoMap;
mutex                                   PhysicalPlan::planGlobalMutex;

PhysicalPlan::PhysicalPlan(vector<string> lossNames) {
    this->lossConsole = NULL;
    SNEW(this->lossConsole, LossConsole, lossNames);
    SASSUME0(this->lossConsole);
}

PhysicalPlan::~PhysicalPlan() {
    for (map<TensorAllocKey, void*>::iterator iter = this->tensorAllocMap.begin();
        iter != this->tensorAllocMap.end(); iter++) {
        void* value = iter->second;
        Data<float>* dataPtr = (Data<float>*)value;
        SDELETE(dataPtr);
    }

    for (map<int, void*>::iterator iter = instanceMap.begin(); iter != instanceMap.end();
        ++iter) {

        int layerID = iter->first;
        void* instancePtr = iter->second;

        int forwardPlanID = LP_FORWARD_PLANID(layerID);
        if (planMap.find(forwardPlanID) != planMap.end()) {
            int layerType = planMap[forwardPlanID].layerType;
            WorkContext::updateLayer(WorkContext::curNetworkID, layerID);
            LayerFunc::destroyLayer(layerType, instancePtr);
        }
    }

    SDELETE(this->lossConsole);
}

void* PhysicalPlan::allocTensorMem(int layerType, void* instancePtr, string tensorName,
    PlanAlloc planAlloc, bool isInput, int index) {

    if (planAlloc.nodeID != SPARAM(NODE_ID)) {
        // allocate GPU memory of other nodes.
        // TODO: 구현!!
        SASSERT(false, "not implemented yet");
    }

    void *tensorPtr;
    if (WorkContext::curBootMode == DeveloperMode ||
        WorkContext::curBootMode == TestMode ||
        WorkContext::curBootMode == ResumeJobMode ||
        WorkContext::curBootMode == SingleJobMode) {
        Data<float>* tensor = NULL;
        SNEW(tensor, Data<float>, tensorName);
        SASSERT0(tensor != NULL);

        tensorPtr = (void*)tensor;
    } else {
        int consumerIdx = Worker::getConsumerIdx(planAlloc.devID);
        TaskAllocTensor* task =
            Worker::addAllocTensorTask(consumerIdx,
            SPARAM(NODE_ID), planAlloc.devID, WorkContext::curThreadID, tensorName);   
       
        ThreadMgmt::signal(ThreadMgmt::getThreadID(ThreadType::TaskConsumer, consumerIdx),
                ThreadEvent::Wakeup);
        ThreadMgmt::wait(WorkContext::curThreadID, 0);

        SASSUME0(task->step == TaskAllocTensorStep::Done);
        tensorPtr = (void*)task->tensorPtr;
        Task::releaseElem(TaskType::AllocTensor, (void*)task); 
    }

    LayerFunc::setInOutTensor(layerType, instancePtr, tensorPtr, isInput, index);
    return tensorPtr;
}

vector<int> PhysicalPlan::getOrderedLayerIDs(string networkID, bool buildTrainActivation) {
    map<string, int> doneTensorMap; 
    map<int, int> doneLayerIDMap;

    vector<int> layerIDs;

    int targetLayerCount = 0;
    for (map<int, PlanAlloc>::iterator iter = this->allocMap.begin();
            iter !=this->allocMap.end(); iter++) {
        
        int layerID = iter->first;
        WorkContext::updateLayer(networkID, layerID);

        if (buildTrainActivation && 
            (SLPROP_BASE(activation) == LayerActivation::TestActivation))
            continue;

        if (!buildTrainActivation && 
            (SLPROP_BASE(activation) == LayerActivation::TrainActivation))
            continue;

        // split layer는 체크하지 않는다. split layer 자체에는 activation 정보를 담기 어렵다.
        // 그래서 특정 split layer가 train용인지 test용인지 구분하기 어렵다. 
        // 게다가 normal layer(split layer가 아닌 레이어)는 그것에 연결된 split layer에 대한
        // 정보가 채워져야 doneLayerIDMap으로 등록이 될 수 있기 때문에 split layer를 체크하지
        // 않아도 동작에는 문제가 없다. 
        if (SLPROP_BASE(id) >= SPARAM(SPLITLAYER_START_LAYERID))
            continue;

        targetLayerCount++;
    }

    while (true) {
        for (map<int, PlanAlloc>::iterator iter = this->allocMap.begin();
            iter !=this->allocMap.end(); iter++) {
            int layerID = iter->first;

            if (doneLayerIDMap.find(layerID) != doneLayerIDMap.end())
                continue;

            WorkContext::updateLayer(networkID, layerID);

            if (buildTrainActivation && 
                (SLPROP_BASE(activation) == LayerActivation::TestActivation))
                continue;

            if (!buildTrainActivation && 
                (SLPROP_BASE(activation) == LayerActivation::TrainActivation))
                continue;

            vector<string> inputs = SLPROP_BASE(input);
            vector<string> outputs = SLPROP_BASE(output);

            bool needTensor = false;

            for (int i = 0; i < inputs.size(); i++) {
                if (doneTensorMap.find(inputs[i]) == doneTensorMap.end()) {
                    needTensor = true;
                    break;
                }
            }

            // input Layer의 경우에 useCompositeModel이 true인 경우에 input layer를 초기화
            // 하면서 input layer의 inputs 텐서 개수가 0에서 늘어날 수 있다. 본 알고리즘은
            // input layer의 inputs 텐서 개수가 0으로 가정하고 만들어졌다. 그것을 위해서 이미
            // 초기화 된 input layer의 경우에는 강제적으로 needTensor를 false로 설정해 준다.
            if (needTensor && 
                (this->instanceMap.find(layerID) != this->instanceMap.end())) {
                // XXX: 현재는 float 데이터 타입만 지원하고 있다... 하지만.. 추후에 generic
                // type을 지원할 수 있도록 변경해야 한다.
                Layer<float>* instancePtr = (Layer<float>*)this->instanceMap[layerID];

                InputLayer<float>* inputLayer = dynamic_cast<InputLayer<float>*>(instancePtr);
                if (inputLayer != NULL)
                    needTensor = false;
            }

            if (needTensor) {
            	//cout << "layerID " << layerID << " needs tensor ... " << endl;
                continue;
            }

            for (int i = 0; i < outputs.size(); i++) {
                if (doneTensorMap.find(outputs[i]) == doneTensorMap.end()) {
                    doneTensorMap[outputs[i]] = 1;
                }
            }

            doneLayerIDMap[layerID] = 1;
            layerIDs.push_back(layerID);
        }

        // 종료 조건을 체크한다.
        int doneTargetCount = 0;
        for (int i = 0; i < layerIDs.size(); i++) {
            if (layerIDs[i] < SPARAM(SPLITLAYER_START_LAYERID))
                doneTargetCount++;
        }

        if (doneTargetCount == targetLayerCount)
            break;
    }

    return layerIDs;
}

void PhysicalPlan::allocateTensorInternal(string networkID, int dopID,
        bool buildTrainActivation) {
    vector<int> orderedIDs = getOrderedLayerIDs(networkID, buildTrainActivation);

    for (int orderedLayerIdx = 0; orderedLayerIdx < orderedIDs.size(); orderedLayerIdx++) {
        int layerID = orderedIDs[orderedLayerIdx];
        SASSUME0(this->allocMap.find(layerID) != this->allocMap.end());
        PlanAlloc planAlloc = this->allocMap[layerID];

        WorkContext::updateLayer(networkID, layerID);

        if (buildTrainActivation && 
                (SLPROP_BASE(activation) == LayerActivation::TestActivation))
            continue;

        if (!buildTrainActivation &&
            (SLPROP_BASE(activation) == LayerActivation::TrainActivation))
            continue;

        vector<string> inputs = SLPROP_BASE(input);
        vector<string> outputs = SLPROP_BASE(output);

        SASSUME0(planMap.find(LP_FORWARD_PLANID(layerID)) != planMap.end());
        int layerType = planMap[LP_FORWARD_PLANID(layerID)].layerType;
        // When you get the layer type, you can use any plan ID that corresponds to the layer
        // ID

        // (0) initialize layer instance
        void* instancePtr;

        if (this->instanceMap.find(layerID) == this->instanceMap.end()) {
            instancePtr = LayerFunc::initLayer(layerType);
            SASSUME0(instancePtr != NULL);
            this->instanceMap[layerID] = instancePtr;

            // (1) allocate input/output tensor
            for (int i = 0; i < inputs.size(); i++) {
                TensorAllocKey key;
                key.tensorAlloc = planAlloc;
                key.tensorName = inputs[i];

                if (tensorAllocMap.find(key) == tensorAllocMap.end()) {
                    void* allocPtr = PhysicalPlan::allocTensorMem(layerType, instancePtr,
                        key.tensorName, key.tensorAlloc, true, i);
                    SASSERT0(allocPtr != NULL);
                    tensorAllocMap[key] = allocPtr;
                } else {
                    void* tensor = tensorAllocMap[key];
                    LayerFunc::setInOutTensor(layerType, instancePtr, tensor, true, i);
                }
            }

            for (int i = 0; i < outputs.size(); i++) {
                TensorAllocKey key;
                key.tensorAlloc = planAlloc;
                key.tensorName = outputs[i];

                if (tensorAllocMap.find(key) == tensorAllocMap.end()) {
                    void* allocPtr = PhysicalPlan::allocTensorMem(layerType, instancePtr,
                        key.tensorName, key.tensorAlloc, false, i);
                    SASSERT0(allocPtr != NULL);
                    tensorAllocMap[key] = allocPtr;
                } else {
                    void* tensor = tensorAllocMap[key];
                    LayerFunc::setInOutTensor(layerType, instancePtr, tensor, false, i);
                }
            }

        } else {
            instancePtr = this->instanceMap[layerID];
        }

        if (WorkContext::curBootMode == DeveloperMode ||
            WorkContext::curBootMode == TestMode ||
            WorkContext::curBootMode == ResumeJobMode ||
            WorkContext::curBootMode == SingleJobMode) {
            SASSERT0(LayerFunc::allocLayerTensors(layerType, instancePtr) == true);
        } else {
            SASSUME0(planAlloc.nodeID == SPARAM(NODE_ID));  // TODO: 다른 노드에 대한건 나중에
            
            int consumerIdx = Worker::getConsumerIdx(planAlloc.devID);
            Worker::addAllocLayerTask(consumerIdx, networkID, dopID, layerID, SPARAM(NODE_ID),
                    planAlloc.devID, WorkContext::curThreadID, layerType, instancePtr);
            
            ThreadMgmt::signal(ThreadMgmt::getThreadID(ThreadType::TaskConsumer, consumerIdx),
                ThreadEvent::Wakeup);
            ThreadMgmt::wait(WorkContext::curThreadID, 0);
        }

        if (SLPROP_BASE(learnable)) {
            if (SLPROP_BASE(donate))
                SLPROP_BASE(donatorID) = SLPROP_BASE(id);

            LearnableLayer<float>* learnableLayer = (LearnableLayer<float>*)instancePtr;

            if (SLPROP_BASE(receive)) {
                SASSERT0(!SLPROP_BASE(donate));
                SASSERT0(SLPROP_BASE(donatorID) >= 0);
                Donator<float>::receive(SLPROP_BASE(donatorID), instancePtr);
            }

            if (SLPROP_BASE(donate)) {
                SASSERT0(!SLPROP_BASE(receive));
                SASSERT0(SLPROP_BASE(donatorID) >= 0);
                Donator<float>::donate(SLPROP_BASE(donatorID), instancePtr);
            }
        }
    }
}

void PhysicalPlan::allocateTensor(string networkID) {
    WorkContext::updateNetwork(networkID);

    unique_lock<mutex> planLock(PhysicalPlan::planGlobalMutex);
    SASSUME0(PhysicalPlan::planGlobalInfoMap.find(networkID) !=
            PhysicalPlan::planGlobalInfoMap.end());
    PlanInfo *planInfo = PhysicalPlan::planGlobalInfoMap[networkID];

    SASSUME0(PhysicalPlan::planGlobalMap.find(networkID) !=
        PhysicalPlan::planGlobalMap.end());
    SASSUME0(PhysicalPlan::planGlobalMap[networkID].size() == planInfo->dopCount);

    vector<PhysicalPlan*> curPPs;
    for (int i = 0; i < planInfo->dopCount; i++) {
        PhysicalPlan* curPP = PhysicalPlan::planGlobalMap[networkID][i];
        curPPs.push_back(curPP);
    }
    planLock.unlock();

    for (int i = 0; i < curPPs.size(); i++) {
        WorkContext::updatePlan(i, true);

        if (SNPROP(useCompositeModel)) {
            curPPs[i]->allocateTensorInternal(networkID, i, true);
            curPPs[i]->allocateTensorInternal(networkID, i, false);
        } else if (SNPROP(status) == NetworkStatus::Train) {
            curPPs[i]->allocateTensorInternal(networkID, i, true);
        } else {
            curPPs[i]->allocateTensorInternal(networkID, i, false);
        }
    }
}

void PhysicalPlan::notifyFinish(int targetPlanID) {
    unique_lock<mutex> planLock(this->planMutex);

    SASSUME(this->depRefMap.find(targetPlanID) != this->depRefMap.end(),
        "There is no ref map for requesting plan ID. targetPlanID=%d", targetPlanID);
    this->depRefMap[targetPlanID] -= 1;
    
    if (this->depRefMap[targetPlanID] == 0) {
        this->readyQueue.push_back(targetPlanID); 
    }

    planLock.unlock();
    SASSUME0(this->depRefMap[targetPlanID] >= 0);
}

void PhysicalPlan::markDone(int planID) {
    unique_lock<mutex> planLock(this->planMutex);

    SASSUME(this->depRefMap.find(planID) != this->depRefMap.end(),
        "There is no ref map for requesting plan ID. planID=%d", planID);

    this->refCount -= 1;
    this->planTypeRCMap[LP_PLANID_TO_PLANTYPE(planID)] -= 1;
    planLock.unlock();

    SASSUME0(this->refCount >= 0);
    SASSUME0(this->planTypeRCMap[LP_PLANID_TO_PLANTYPE(planID)] >= 0);
}

void PhysicalPlan::markFinish(string networkID, int dopID, int planID) {
    WorkContext::updateNetwork(networkID);
    WorkContext::updatePlan(dopID, true);

    PhysicalPlan* pp = PhysicalPlan::getCurPhysicalPlan();
    PlanDef planDef = pp->planMap[planID];

    for (int i = 0; i < planDef.notifyList.size(); i++) {
        int targetPlanID = planDef.notifyList[i];
        pp->notifyFinish(targetPlanID);
    }
    pp->markDone(planID);
}

void PhysicalPlan::saveNetwork(bool checkCond) {
    if (checkCond) {
        bool saveNetwork = false;

        if ((SNPROP(saveInterval) != 0) &&
            ((SNPROP(iterations) % SNPROP(saveInterval)) == 0)) {
            saveNetwork = true;
        } 

        if (!saveNetwork)
            return;
    }

    string networkID = WorkContext::curNetworkID;
    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->handleIntervalSaveParams(SNPROP(iterations));
}

void PhysicalPlan::loadNetwork() {
    if (SNPROP(loadPath) == "") {
        if ((SNPROP(loadPathForTest) == "") ||
            (SNPROP(status) == NetworkStatus::Train) ||
            (SNPROP(useCompositeModel)))
            return;
    }
    
    string networkID = WorkContext::curNetworkID;
    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->load();

}

float PhysicalPlan::calcLoss() {
    float avgLoss = 0.0;

    if (SNPROP(status) == NetworkStatus::Test)
        return 0.0;

    if (SNPROP(lossLayer).size() == 0)
        return 0.0;

    for (int i = 0; i < SNPROP(lossLayer).size(); i++) {
        string lossLayerName = SNPROP(lossLayer)[i];
        Network<float>* network = Network<float>::getNetworkFromID(WorkContext::curNetworkID);
        Layer<float>* layer = network->findLayer(lossLayerName);
        SASSERT(layer != NULL, "Could not find loss layer: %s", lossLayerName.c_str());
        LossLayer<float>* lossLayer = (LossLayer<float>*)layer;
        avgLoss += (float)lossLayer->cost();
    }

    return avgLoss / (float)(SNPROP(lossLayer).size());
}

void PhysicalPlan::logLoss() {
    if (SNPROP(testInterval) == 0)
        return;

    if (SNPROP(status) == NetworkStatus::Test)
        return;

    for (int i = 0; i < SNPROP(lossLayer).size(); i++) {
        string lossLayerName = SNPROP(lossLayer)[i];
        Network<float>* network = Network<float>::getNetworkFromID(WorkContext::curNetworkID);
        Layer<float>* layer = network->findLayer(lossLayerName);
        LossLayer<float>* lossLayer = (LossLayer<float>*)layer;

        lossConsole->addValue(i, (float)lossLayer->cost());
    }

    if (SNPROP(iterations) % SNPROP(testInterval) == 0) {
        lossConsole->printLoss(stdout);
        lossConsole->clear();
    }
}

bool PhysicalPlan::generatePlan(bool genNextMiniBatch, bool genPlanOnly) {
    // (1) mini batch를 다 돌았는지 확인한다.
    // FIXME: plan lock의 범위가 좀 넓다.. 최적화 고민해보자.
    unique_lock<mutex> planLock(this->planMutex);
    if (this->refCount > 0) {
        planLock.unlock();
        return true;
    }

    // FIXME: I/O를 sync로 진행하고 있다. async로 동작하도록 해야 효율적으로 동작한다.
    //        하지만, async로 했을때에 자원보호 부분이 복잡해진다. 
    //        추후에 좀 더 세밀하게 생각해서 수정하도록 하자.
    string networkID = WorkContext::curNetworkID;
    Network<float>* network = Network<float>::getNetworkFromID(networkID);

    if (!genPlanOnly && network->getMeasureInserted() && 
        SNPROP(status) == NetworkStatus::Train) {
        MeasureEntry* measureEntry = MeasureManager::getMeasureEntry(networkID); 
        SASSUME0(measureEntry != NULL);

        for (int i = 0; i < SNPROP(measureLayer).size(); i++) {
            string measureLayerName = SNPROP(measureLayer)[i];
            Layer<float>* layer = network->findLayer(measureLayerName);

            MeasureLayer<float>* measureLayer = 
                dynamic_cast<MeasureLayer<float>*>(layer);
            
            if (measureLayer != NULL) {
                float measureVal = measureLayer->measure();
                if (measureVal != measureVal) // NaN case
                    measureVal = 0.0;
                measureEntry->getAddBuffer()[i] = measureVal;
            } else {
                LossLayer<float>* lossLayer = dynamic_cast<LossLayer<float>*>(layer);
                SASSUME0(lossLayer != NULL);
                measureEntry->getAddBuffer()[i] = lossLayer->cost();
            }
        }

        measureEntry->addData(measureEntry->getAddBuffer());
        MeasureManager::releaseMeasureEntry(networkID);
    }

    bool saveNetwork = false;
    if (!genPlanOnly) {
        float currLoss = calcLoss();
        // loss에 NaN 값이 있는지 체크한다.
        if (SPARAM(STOP_TRAIN_WHEN_GOT_NAN_LOSS)) {
            if (currLoss != currLoss) {
                planLock.unlock();
                COLD_LOG(ColdLog::WARNING, true,
                    "training network(id=%s) is stopped due to NaN loss at epoch=%d",
                    WorkContext::curPlanInfo->networkID.c_str(),
                    WorkContext::curPlanInfo->curEpochIndex);
                return false;
            }
        }

        if (network->getStop()) {
            planLock.unlock();
            COLD_LOG(ColdLog::WARNING, true,
                "training network(id=%s) is stopped due to user request",
                WorkContext::curPlanInfo->networkID.c_str());
            PhysicalPlan::saveNetwork(false);
            return false;
        }

        // (2) plan info의 curMiniBatchIndex, curEpochIndex를 갱신한다.
        // FIXME: planInfoLock 범위가 너무 크다!!!
        unique_lock<mutex> planInfoLock(WorkContext::curPlanInfo->planMutex);

        if (WorkContext::curPlanInfo->curMiniBatchIndex == 
                WorkContext::curPlanInfo->miniBatchCount - 1) {
            WorkContext::curPlanInfo->curMiniBatchIndex = 0;
            WorkContext::curPlanInfo->curEpochIndex += 1;
        } else {
            WorkContext::curPlanInfo->curMiniBatchIndex += 1;
        }

        if ((SNPROP(saveInterval) != 0) && (SNPROP(iterations) > 0) &&
            ((SNPROP(iterations) % SNPROP(saveInterval)) == 0)) {
            saveNetwork = true;
        }

        // best loss를 가지고 있는 네트워크를 저장한다.
        network->handleBestLoss(currLoss, SNPROP(iterations));

        if (WorkContext::curPlanInfo->curEpochIndex >= WorkContext::curPlanInfo->epochCount) {
            WorkContext::curPlanInfo->curEpochIndex -= 1;
            WorkContext::curPlanInfo->curMiniBatchIndex =
                WorkContext::curPlanInfo->miniBatchCount - 1;

            planInfoLock.unlock();
            planLock.unlock();

            if (SNPROP(saveInterval) != 0)
                PhysicalPlan::saveNetwork(false);
                
            logLoss();

            SNPROP(iterations) += 1;
            return false;
        }
        planInfoLock.unlock();
    }

    // (3) 초기화를 수행한다.
    if (genNextMiniBatch) {
        this->refCount = 0;
        for (int i = 0 ; i < PlanType::PLANTYPE_MAX; i++) {
            this->planTypeRCMap[(PlanType)i] = 0;
        }
        this->readyQueue = {};
        for (map<int, PlanDef>::iterator it = planMap.begin(); it != planMap.end(); ++it) {
            int key = it->first;
            PlanDef value = it->second;
          
            depRefMap[key] = value.depCount;

            if (value.depCount == 0) {
                readyQueue.push_back(key);
            }

            this->refCount += 1;
            SASSUME0(value.planType < PlanType::PLANTYPE_MAX);
            this->planTypeRCMap[value.planType] += 1;
        }
    }

    planLock.unlock();

    if (!genPlanOnly) {
        logLoss();

        if (saveNetwork)
            PhysicalPlan::saveNetwork(false);

        SNPROP(iterations) += 1;
        return true;
    } else {
        return false;
    }
}

void PhysicalPlan::reset() {
    this->refCount = 0;
    unique_lock<mutex> planLock(this->planMutex);
    this->readyQueue.clear();
    
    for (map<int, PlanDef>::iterator it = planMap.begin(); it != planMap.end(); ++it) {
        int key = it->first;
        PlanDef value = it->second;
      
        depRefMap[key] = value.depCount;

        if (value.depCount == 0) {
            readyQueue.push_back(key);
        }

        this->refCount += 1;
        SASSUME0(value.planType < PlanType::PLANTYPE_MAX);
        this->planTypeRCMap[value.planType] += 1;
    }
}

void PhysicalPlan::runLayer(int planID, bool inference) {
    // (1) set context
    int layerID = LP_PLANID_TO_LAYERID(planID);
    WorkContext::updateLayer(WorkContext::curNetworkID, layerID);
    PlanType planType = LP_PLANID_TO_PLANTYPE(planID);

    // FIXME: 나름 핫 코드영역인데 이렇게 자주 맵을 뒤지면 성능에 안좋다. 수정필요!!
    SASSUME0(this->planMap.find(planID) != this->planMap.end());
    int layerType = this->planMap[planID].layerType;

    // (1.5) activation 속성을 바탕으로 runLayer 여부를 결정한다.
    bool runLayer = true;
    if (SLPROP_BASE(activation) != LayerActivation::AllActivation) {
        if (inference && SLPROP_BASE(activation) == LayerActivation::TrainActivation)
            runLayer = false;

        if (!inference && SLPROP_BASE(activation) == LayerActivation::TestActivation)
            runLayer = false;
    }

    // (2) run layer
    bool doMarkFinish = true;
    if (runLayer && (!inference || (planType == PLANTYPE_FORWARD))) {
        PlanInfo* planInfo = WorkContext::curPlanInfo;

        STDOUT_COND_BLOCK(SPARAM(PRINT_RUNLAYER_LOG), 
        cout << "Epoch : " << planInfo->curEpochIndex << ", minibatch : " << 
            planInfo->curMiniBatchIndex << " run layer (planID=" << planID << ")" << endl);

        SASSUME0(this->instanceMap.find(layerID) != this->instanceMap.end());
        void* instancePtr = this->instanceMap[layerID];


        SASSUME0(planType < PLANTYPE_MAX);
        if (planType == PLANTYPE_FORWARD) {
            LayerFunc::runForward(layerType, instancePtr, planInfo->curMiniBatchIndex);
        } else if (planType == PLANTYPE_BACKWARD) {
            LayerFunc::runBackward(layerType, instancePtr);
        } else {
            SASSUME0(planType == PLANTYPE_UPDATE);
            LayerFunc::learn(layerType, instancePtr);
            doMarkFinish = false;       // update는 내부적으로 mark finish한다.
        }
    }

    // (3) mark
    if (!doMarkFinish)
        return;

    PlanDef *planDef = &WorkContext::curPhysicalPlan->planMap[planID];
    for (int i = 0; i < planDef->notifyList.size(); i++) {
        int targetPlanID = planDef->notifyList[i];
        notifyFinish(targetPlanID);
    }
    markDone(planID);
}

bool PhysicalPlan::runPlan(int layerID, PlanType planType, bool inference) {
    int targetPlanID;


    if (planType == PLANTYPE_FORWARD) {
        targetPlanID = LP_FORWARD_PLANID(layerID);
    } else if (planType == PLANTYPE_BACKWARD) {
        targetPlanID = LP_BACKWARD_PLANID(layerID);
    } else if (planType == PLANTYPE_UPDATE) {
        targetPlanID = LP_UPDATE_PLANID(layerID);
    } else {
        SASSERT(false, "invalid plan type. plan type=%d", (int)planType);
    }

    bool found = false;
    unique_lock<mutex> planLock(this->planMutex);
    for (list<int>::iterator iter = this->readyQueue.begin(); iter != this->readyQueue.end();
        iter++) {
        int value = (*iter);
        if (value == targetPlanID) {
            found = true;
            this->readyQueue.erase(iter);
            break;
        }
    }

    planLock.unlock();

    if (!found)
        return false;

    runLayer(targetPlanID, inference);
    return true;
}

bool PhysicalPlan::runPlan(bool inference) {
    unique_lock<mutex> planLock(this->planMutex);
    
    if (this->readyQueue.size() == 0) {
        return false;
    }

    int planID = this->readyQueue.front();
    this->readyQueue.pop_front();
    planLock.unlock();

    SASSUME0(this->planMap.find(planID) != this->planMap.end());

    runLayer(planID, inference);
    return true;
}

bool PhysicalPlan::runPlan(PlanType planType, bool inference) {
    bool found = false;
    int targetPlanID;
    unique_lock<mutex> planLock(this->planMutex);
    for (list<int>::iterator iter = this->readyQueue.begin(); iter != this->readyQueue.end();
        iter++) {
        int value = (*iter);
        if (planType == LP_PLANID_TO_PLANTYPE(value)) {
            found = true;
            targetPlanID = value;
            this->readyQueue.erase(iter);
            break;
        }
    }

    planLock.unlock();

    if (!found)
        return false;

    runLayer(targetPlanID, inference);
}

void PhysicalPlan::insertPlan(string networkID, vector<PhysicalPlan*> pMap, PlanInfo *pInfoMap) {
    unique_lock<mutex> planLock(PhysicalPlan::planGlobalMutex);
    SASSERT0(PhysicalPlan::planGlobalMap.find(networkID) == 
            PhysicalPlan::planGlobalMap.end());
    PhysicalPlan::planGlobalMap[networkID] = {};
    for (int i = 0; i < pMap.size(); i++) {
        PhysicalPlan::planGlobalMap[networkID].push_back(pMap[i]);
    }

    SASSERT0(PhysicalPlan::planGlobalInfoMap.find(networkID) ==
            PhysicalPlan::planGlobalInfoMap.end());

    PhysicalPlan::planGlobalInfoMap[networkID] = pInfoMap;
}

void PhysicalPlan::removePlan(string networkID) {
    WorkContext::updateNetwork(networkID);

    unique_lock<mutex> planLock(PhysicalPlan::planGlobalMutex);
    SASSERT0(PhysicalPlan::planGlobalMap.find(networkID) != 
            PhysicalPlan::planGlobalMap.end());

    // XXX: 이거 task consumer에서 처리해야 한다.
    vector<PhysicalPlan*>::iterator iter;
    for (iter = PhysicalPlan::planGlobalMap[networkID].begin(); 
            iter != PhysicalPlan::planGlobalMap[networkID].end(); ) {
        PhysicalPlan* pp = (PhysicalPlan*)(*iter);
        WorkContext::updatePlan(pp->dopID, false);
        SDELETE(pp);

        iter = PhysicalPlan::planGlobalMap[networkID].erase(iter);
    }

    SASSERT0(PhysicalPlan::planGlobalInfoMap.find(networkID) !=
            PhysicalPlan::planGlobalInfoMap.end());

    PlanInfo* deletePlanInfo = PhysicalPlan::planGlobalInfoMap[networkID];
    SDELETE(deletePlanInfo);
    PhysicalPlan::planGlobalInfoMap.erase(networkID);
}

PhysicalPlan* PhysicalPlan::getCurPhysicalPlan() {
    return WorkContext::curPhysicalPlan;
}

void PhysicalPlan::setCurPlanInfo(string networkID) {
    unique_lock<mutex> planLock(PhysicalPlan::planGlobalMutex);
    if (PhysicalPlan::planGlobalInfoMap.find(networkID) != 
        PhysicalPlan::planGlobalInfoMap.end()) {
        WorkContext::curPlanInfo = PhysicalPlan::planGlobalInfoMap[networkID];
    }
}

void PhysicalPlan::setCurPlan(string networkID, int dopID, bool acquireLock) {

    if (acquireLock)
        PhysicalPlan::planGlobalMutex.lock();

    SASSERT0(PhysicalPlan::planGlobalMap.find(networkID) != 
            PhysicalPlan::planGlobalMap.end());

    SASSUME0(dopID < PhysicalPlan::planGlobalMap[networkID].size());
    WorkContext::curPhysicalPlan = PhysicalPlan::planGlobalMap[networkID][dopID];

    if (acquireLock)
        PhysicalPlan::planGlobalMutex.unlock();
}

int PhysicalPlan::getDOPCount(string networkID) {
    unique_lock<mutex> planInfoLock(PhysicalPlan::planGlobalMutex);
    SASSUME0(PhysicalPlan::planGlobalInfoMap.find(networkID) !=
            PhysicalPlan::planGlobalInfoMap.end());
    PlanInfo* planInfo = PhysicalPlan::planGlobalInfoMap[networkID];
    int dopCount = planInfo->dopCount;
    planInfoLock.unlock();
    return dopCount;
}

void PhysicalPlan::setCurProgress(string networkID, int iterNum) {
    unique_lock<mutex> planInfoLock(PhysicalPlan::planGlobalMutex);
    SASSUME0(PhysicalPlan::planGlobalInfoMap.find(networkID) !=
            PhysicalPlan::planGlobalInfoMap.end());
    PlanInfo* planInfo = PhysicalPlan::planGlobalInfoMap[networkID];
    planInfo->curEpochIndex = iterNum / planInfo->miniBatchCount;
    planInfo->curMiniBatchIndex = iterNum % planInfo->miniBatchCount;
}

void* PhysicalPlan::getTensor(int nodeID, int devID, string tensorName) {
    TensorAllocKey key;
    key.tensorAlloc.nodeID = nodeID;
    key.tensorAlloc.devID = devID;
    key.tensorName = tensorName;
    if (tensorAllocMap.find(key) == tensorAllocMap.end())
        return NULL;
    else
        return tensorAllocMap[key];
}
