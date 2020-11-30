/*
 * Network.cpp
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#include <stdlib.h>

#include <vector>
#include <map>
#include <cfloat>
#include <string>
#include <iostream>
#include <limits>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/lexical_cast.hpp>

#include "DataSet.h"
#include "BaseLayer.h"
#include "SoftmaxWithLossLayer.h"
#include "LossLayer.h"
#include "Util.h"
#include "Worker.h"
#include "Perf.h"
#include "StdOutLog.h"
#include "Network.h"
#include "SysLog.h"
#include "DebugUtil.h"
#include "WorkContext.h"
#include "PhysicalPlan.h"
#include "PlanOptimizer.h"
#include "PropMgmt.h"
#include "LearnableLayer.h"
#include "LogicalPlan.h"
#include "MeasureManager.h"
#include "FileMgmt.h"
#include "MemoryMgmt.h"
#include "PlanParser.h"

using namespace std;
using namespace boost::uuids;

extern const char*  LAONSILL_HOME_ENVNAME;

template<typename Dtype>
map<string, Network<Dtype>*>   Network<Dtype>::networkIDMap;
template<typename Dtype>
mutex Network<Dtype>::networkIDMapMutex;

template <typename Dtype>
Network<Dtype>::Network() {
    random_generator gen;
    uuid id = gen();
    this->networkID = to_string(id);
    this->adhocRunRefCount = 0;

    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    Network<Dtype>::networkIDMap[this->networkID] = this;
    this->isLoaded = false;
    this->isBuilt = false;
    this->isMeasureInserted = false;
    this->isNeedStop = false;

    this->bestLoss = numeric_limits<float>::max();
    this->bestSavedParamPath = "";

    // train 정보를 관리하는 파일 포인터를 얻는다.
    string trainFilePath = string(getenv(LAONSILL_HOME_ENVNAME)) + "/param/" +
            this->networkID + ".train";
    this->trainFP = fopen(trainFilePath.c_str(), "w+");
    SASSERT0(this->trainFP != NULL);
}

template <typename Dtype>
Network<Dtype>::~Network() {
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    unique_lock<mutex> adhocRunLock(this->adhocRunMutex);

    // FIXME: 이 과정 자체에서 큰 부하가 있을것으로 보이진 않아서 나이브하게 구현하였다.
    // 추후에 이 부분에 부하가 커진다면 (즉, 빨리 리소스를 정리하지 않으면 안되는 상황인 경우)
    // 정리쓰레드를 따로 만들어서 refcount가 0이 된 순간에 빨리 자원을 해지하도록 수정하자.
    while (true) {
        if (this->adhocRunRefCount == 0) {
            adhocRunLock.unlock();
            break;
        } else {
            adhocRunLock.unlock();
            lock.unlock();
            usleep(SPARAM(NETWORK_ADHOC_REFCOUNT_CHECKTIME_USEC));
            lock.lock();
            adhocRunLock.lock();
        }
    }

    Network<Dtype>::networkIDMap.erase(this->networkID);
    lock.unlock();

    if (this->isMeasureInserted) {
        MeasureManager::removeEntry(this->networkID);
    }

    if (this->trainFP != NULL)
        fclose(this->trainFP);

    if (SPARAM(USE_MEMORY_DUMP_WHEN_NETWORK_DESTROYED)) {
        MemoryMgmt::dump(MemoryMgmtSortOptionIndex, true);
        MemoryMgmt::dump(MemoryMgmtSortOptionSize, true);
    }
}

template<typename Dtype>
bool Network<Dtype>::addAdhocRun(std::string networkID) {
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    if (Network<Dtype>::networkIDMap.find(networkID) == Network<Dtype>::networkIDMap.end()) {
        return false;
    }

    Network<Dtype>* network = Network<Dtype>::networkIDMap[networkID];

    unique_lock<mutex> adhocRunLock(network->adhocRunMutex);
    network->adhocRunRefCount++;
    adhocRunLock.unlock(); 
    return true;
}

template<typename Dtype>
void Network<Dtype>::removeAdhocRun(std::string networkID) {
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);

    unique_lock<mutex> adhocRunLock(network->adhocRunMutex);
    SASSUME0(network->adhocRunRefCount > 0);
    network->adhocRunRefCount--;
    adhocRunLock.unlock();
}

template<typename Dtype>
void Network<Dtype>::init() {
}

template<typename Dtype>
Network<Dtype>* Network<Dtype>::getNetworkFromID(string networkID) {
    Network<Dtype>* network;
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    network = Network<Dtype>::networkIDMap[networkID];
    lock.unlock();
    return network;
}

template<typename Dtype>
void Network<Dtype>::stopNetwork(string networkID) {
    Network<Dtype>* network;
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);

    if (Network<Dtype>::networkIDMap.find(networkID) == Network<Dtype>::networkIDMap.end()) {
        lock.unlock();
        COLD_LOG(ColdLog::WARNING, true, "network already stopped. network ID=%s",
                networkID.c_str());
        return;
    }

    network = Network<Dtype>::networkIDMap[networkID];
    network->setStop();
}

template <typename Dtype>
void Network<Dtype>::run_with_timer( bool inference) {
    struct timespec startTime;
    SPERF_START(NETWORK_TRAINING_TESTTIME, &startTime);
	run(inference);

    SPERF_END(NETWORK_TRAINING_TESTTIME, startTime, SNPROP(epochs));
    STDOUT_BLOCK(cout << "Total Training Time : " << SPERF_TIME(NETWORK_TRAINING_TESTTIME)
                    << endl;);
}

template<typename Dtype>
void Network<Dtype>::build(int epochs) {
    SASSERT0(this->isLoaded);
        
    WorkContext::updateNetwork(this->networkID); 

    if (epochs > 0)
        SNPROP(epochs) = epochs;

    if (!this->isMeasureInserted && (SNPROP(measureLayer).size() > 0)) {
        MeasureManager::insertEntry(this->networkID, SNPROP(measureLayer));
        this->isMeasureInserted = true;
    }

    PlanOptimizer::buildPlans(networkID);
}

template<typename Dtype>
void Network<Dtype>::reset() {
    SASSERT0(this->isLoaded);

    WorkContext::updateNetwork(this->networkID); 

    PlanInfo* planInfo = WorkContext::curPlanInfo;
    planInfo->curEpochIndex = 0;
    planInfo->curMiniBatchIndex = 0;
    planInfo->doneCount = 0;
    SNPROP(iterations) = 0;
    
    for (int i = 0; i < planInfo->dopCount; i++) {
        WorkContext::updatePlan(i, true);
        PhysicalPlan* pp = PhysicalPlan::getCurPhysicalPlan();
        pp->reset();
    }
}

template<typename Dtype>
void Network<Dtype>::run(bool inference) {
    SASSERT0(this->isLoaded);
    PlanOptimizer::runPlan(this->networkID, inference);
}

template<typename Dtype>
bool Network<Dtype>::getTrainInfo(string networkID, string &networkDef, 
        vector<pair<int, string>> &params) {

    string trainFilePath = 
        string(getenv(LAONSILL_HOME_ENVNAME)) + "/param/" + networkID + ".train";

    ifstream input(trainFilePath.c_str());
    string line;

    if (!input.is_open())
        return false;
   
    bool parseParamPath = false;

    string networkDefStr = "";
    while (input.good()) {
        getline(input, line);

        if (!parseParamPath) {
            if (line.find("======") != string::npos) {
                parseParamPath = true;
            } else {
                networkDefStr += line + "\n";
            }
            continue;
        }

        size_t commaPos;
        commaPos = line.find(",");
        if (commaPos == string::npos)
            continue;
      
        string iterNumStr = line.substr(0, commaPos);

        // XXX: 김종헌 연구원 추가 
        // iteration number로 시작하지 않고 best로 시작하는 라인을 처리하기 위함
        // "best(105),/path" ...
        const string bestPrefix = "best(";
        const int bestPrefixSize = bestPrefix.size();
        if (!iterNumStr.compare(0, bestPrefixSize, bestPrefix)) {
            iterNumStr = iterNumStr.substr(bestPrefixSize, 
                    iterNumStr.size() - (bestPrefixSize + 1));
        }

        int iterNum = stoi(iterNumStr);
        string paramPathStr = line.substr(commaPos + 1, line.length() - 1);

        pair<int, string> p = make_pair(iterNum, paramPathStr);
        params.push_back(p);
    }

    input.close();

    networkDef = networkDefStr;
    return true;
}

template<typename Dtype>
bool Network<Dtype>::getMeasureInfo(string networkID, int targetIter,
        vector<pair<int, vector<float>>> &measures) {

    string measureFilePath = 
        string(getenv(LAONSILL_HOME_ENVNAME)) + "/measure/" + networkID + ".measure";

    ifstream input(measureFilePath.c_str());
    string line;

    if (!input.is_open())
        return false;

    try {
        string networkDefStr = "";
        while (input.good()) {
            if (!getline(input, line))
                break;

            istringstream ss(line);
            vector<string> record;

            while (ss) {
                string s;
                if (!getline(ss, s, ','))
                    break;
                record.push_back(s);
            }

            if (record.size() >= 2) {
                int iter = atoi(record[0].c_str());
                if (iter > targetIter)
                    continue;

                vector<float> data;

                for (int i = 1; i < record.size(); i++) {
                    float floatData = atof(record[i].c_str());
                    data.push_back(floatData);
                }

                measures.push_back(make_pair(iter, data));
            }
        }
    } catch (exception const &e) {
        SASSERT("parsing measure file(measureFilePath=%s) is failed. reason=%s.",
            measureFilePath.c_str(), e.what());
    }

    input.close();

    return true;
}

template<typename Dtype>
string Network<Dtype>::createResumeNetwork(string networkID, string paramFileName,
        bool keepHistory) {
    // TODO: 주어진 네트워크 아이디에 대한 네트워크가 살아 있는지 체크 필요.

    string networkDef;
    vector<pair<int, string>> params;

    bool canGetTrainInfo = Network<Dtype>::getTrainInfo(networkID, networkDef, params);
    if (!canGetTrainInfo)
        return "";

    int iterNum;
    string realParamFilePath;
    string newNetworkID;
    vector<pair<int, vector<float>>> measures;

    if (params.empty()) {
        newNetworkID = PlanParser::loadNetworkByJSONString(networkDef, "", 0);
    } else {
        if (paramFileName == "") {
            iterNum = params.back().first;
            realParamFilePath = params.back().second;
        } else {
            bool paramFound = false;

            for (int i = 0; i < params.size(); i++) {
                if (params[i].second.find(paramFileName) != string::npos) {
                    iterNum = params[i].first;
                    realParamFilePath = params[i].second;
                    paramFound = true;
                    break;
                }
            }

            if (!paramFound)
                return "";
        }

        SASSERT0(realParamFilePath != "");

        ifstream checkFile(realParamFilePath);
        if (!checkFile.good()) {
            checkFile.close();
            return "";
        }
        checkFile.close();

        if (keepHistory) {
            bool canGetMeasureInfo = Network<Dtype>::getMeasureInfo(networkID, iterNum, measures);
            if (canGetMeasureInfo == false) {
                COLD_LOG(ColdLog::ERROR, true, "can not get measure files of network ID=%s",
                        networkID.c_str());
                return "";
            }
        }

        newNetworkID = PlanParser::loadNetworkByJSONString(networkDef, realParamFilePath,
                iterNum);
    }

    WorkContext::updateNetwork(newNetworkID);

    Network<float>* newNetwork = Network<float>::getNetworkFromID(newNetworkID);
    if (!newNetwork->getMeasureInserted() && (SNPROP(measureLayer).size() > 0)) {
        MeasureManager::insertEntry(newNetworkID, SNPROP(measureLayer));
        newNetwork->setMeasureInserted();
    }

    if (keepHistory) {
        newNetwork->logTrainHistory(params);
        if (newNetwork->getMeasureInserted())
            newNetwork->logMeasureHistory(measures);
    }

    return newNetworkID;
}

template<typename Dtype>
void Network<Dtype>::runAdhoc(string inputLayerName, int channel, int height, int width,
        float* imageData) {
    SASSERT0(this->isLoaded);
    PlanOptimizer::runAdhocPlan(this->networkID, inputLayerName, channel, height, width,
            imageData);
}

template<typename Dtype>
void Network<Dtype>::runPlanType(PlanType planType, bool inference) {
    SASSERT0(this->isLoaded);
    PlanOptimizer::runPlanByType(this->networkID, planType, inference);
}

template<typename Dtype>
void Network<Dtype>::runMiniBatch(bool inference, int miniBatchIdx) {
    SASSERT0(this->isLoaded);

    WorkContext::updateNetwork(this->networkID); 
    WorkContext::updatePlan(WorkContext::curDOPID, true);

    PlanInfo* planInfo = WorkContext::curPlanInfo;

    SASSERT0(miniBatchIdx >= 0);

    planInfo->curMiniBatchIndex = miniBatchIdx;
    planInfo->curEpochIndex = 0;
    planInfo->miniBatchCount = miniBatchIdx + 1;
    planInfo->epochCount = 1;
    planInfo->doneCount = 0;

    for (int i = 0; i < planInfo->dopCount; i++) {
        WorkContext::updatePlan(i, true);
        PhysicalPlan* pp = PhysicalPlan::getCurPhysicalPlan();
        pp->reset();
    }

    PlanOptimizer::runPlan(this->networkID, inference);
}

template<typename Dtype>
void Network<Dtype>::save(string path) {
	// save learned params
	ofstream paramOfs(path.c_str(), ios::out | ios::binary);

	uint32_t numParams = 0;
    WorkContext::updateNetwork(this->networkID);
    WorkContext::updatePlan(0, true);   // 아무 dopID에서 가져가도 상관없을꺼 같다.
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;
    for (map<int, void*>::iterator iter = pp->instanceMap.begin();
        iter != pp->instanceMap.end(); iter++) {
        int layerID = iter->first;
        void* instancePtr = iter->second;

        WorkContext::updateLayer(this->networkID, layerID);
        if (!SLPROP_BASE(learnable))
            continue;

        LearnableLayer<Dtype>* ll = (LearnableLayer<Dtype>*)instancePtr;
        numParams += ll->numParams();
    }

	paramOfs.write((char*)&numParams, sizeof(uint32_t));
    for (map<int, void*>::iterator iter = pp->instanceMap.begin();
        iter != pp->instanceMap.end(); iter++) {
        int layerID = iter->first;
        void* instancePtr = iter->second;

        WorkContext::updateLayer(this->networkID, layerID);
        if (!SLPROP_BASE(learnable))
            continue;

        LearnableLayer<Dtype>* ll = (LearnableLayer<Dtype>*)instancePtr;
        ll->saveParams(paramOfs);
    }

	paramOfs.close();

    if (SPARAM(PRINT_PARAMLOG_AFTER_NETWORKSAVE)) {
        //DebugUtil<Dtype>::printNetworkParams(stderr, "network save result",
        //    this->networkID, 0);
        DebugUtil<Dtype>::printNetworkEdges(stderr, "network save result",
            this->networkID, 0);
    }
}

template <typename Dtype>
string Network<Dtype>::save() {
    string path;
	if (SNPROP(savePathPrefix) == "") {
        path = string(getenv(LAONSILL_HOME_ENVNAME)) + "/param/" +
            this->networkID + "_" +
            to_string(SNPROP(iterations)) + ".param";
    } else {
        path = SNPROP(savePathPrefix) + + "_" + to_string(SNPROP(iterations)) + ".param";
    }

    save(path);
    return path;
}

template<typename Dtype>
void Network<Dtype>::handleIntervalSaveParams(int iterNum) {
    if (this->intervalSavedParamPathQueue.size() == SNPROP(keepSaveIntervalModelCount)) {
        string removeParamPath = this->intervalSavedParamPathQueue.front();
        this->intervalSavedParamPathQueue.pop();
        FileMgmt::removeFile(removeParamPath.c_str());
    }

    string newParamPath = this->save();
    this->intervalSavedParamPathQueue.push(newParamPath);

    logTrainFile(to_string(iterNum) + "," + newParamPath);
}

template<typename Dtype>
void Network<Dtype>::handleBestLoss(float loss, int iterNum) {
    if (!SNPROP(keepSaveBestModel))
        return;

    if (SNPROP(keepSaveBestModelStartIterNum) > iterNum)
        return;

    if (loss > this->bestLoss)
        return; 

    this->bestLoss = loss;

    // XXX: remove file 하고 나서 best model을 저장하는 순간에 서버가 죽으면 좀 난감하다.
    //      이 부분에 대한 고려가 필요하다.
    string newParamPath = string(getenv(LAONSILL_HOME_ENVNAME)) + "/param/" +
        this->networkID + "_best_" + to_string(iterNum) + ".param";

    this->save(newParamPath);

    if (this->bestSavedParamPath != "")
        FileMgmt::removeFile(this->bestSavedParamPath.c_str()); 
    this->bestSavedParamPath = newParamPath;

    logTrainFile("best(" + to_string(iterNum) + ")," + newParamPath);
}

template <typename Dtype>
void Network<Dtype>::load(string path) {
    if (path == "")
        return;

    ifstream ifs(path, std::ios::in | std::ios::binary);

    if (!ifs.is_open())
    	STDOUT_LOG("[ERROR] Could not open file: %s", path.c_str());
    SASSERT(ifs.is_open(), "Could not open file: %s", path.c_str());

    // TODO : 반드시 구현 필요
	// load data list from model file
	map<std::string, Data<float>*> dataMap;

    uint32_t numData;
    ifs.read((char*)&numData, sizeof(uint32_t));

    Data<float>::printConfig = true;
    cout << "Load Pretrained Weights ... ----------" << endl;
    for (uint32_t j = 0; j < numData; j++) {
        Data<float>* data = NULL;
        SNEW(data, Data<float>, "", true);
        SASSUME0(data != NULL);
        data->load(ifs);

        if (data)
            data->print();

        string dataName;
        dataName = data->_name;

        map<string, Data<float>*>::iterator it;
        it = dataMap.find(dataName);
        if (it != dataMap.end()) {
            cout << dataName << " overwrites ... " << endl;
            SDELETE(it->second);
        }

        dataMap[dataName] = data;
        cout << data->_name << " is set to " << dataName << endl;
    }
    cout << "--------------------------------------" << endl;
    Data<float>::printConfig = false;
    ifs.close();

    WorkContext::updateNetwork(this->networkID);
    WorkContext::updatePlan(0, true);   // 아무 dopID에서 가져가도 상관없을꺼 같다.
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;
    for (map<int, void*>::iterator iter = pp->instanceMap.begin();
        iter != pp->instanceMap.end(); iter++) {
        int layerID = iter->first;
        void* instancePtr = iter->second;

        WorkContext::updateLayer(this->networkID, layerID);
        if (!SLPROP_BASE(learnable))
            continue;

        LearnableLayer<Dtype>* ll = (LearnableLayer<Dtype>*)instancePtr;
        ll->loadParams(dataMap);
    }

	map<std::string, Data<float>*>::iterator it;
	for (it = dataMap.begin(); it != dataMap.end(); it++)
		SDELETE(it->second);
	dataMap.clear();

    if (SPARAM(PRINT_PARAMLOG_AFTER_NETWORKLOAD)) {
        DebugUtil<Dtype>::printNetworkParams(stderr, "network load result",
            this->networkID, 0);
    }
}

template <typename Dtype>
void Network<Dtype>::load() {
    if ((SNPROP(status) == NetworkStatus::Test) &&
        (SNPROP(loadPathForTest) != "") &&
        !SNPROP(useCompositeModel)) {
        load(SNPROP(loadPathForTest));
    } else {
        load(SNPROP(loadPath));
    }
}

template <typename Dtype>
Layer<Dtype>* Network<Dtype>::findLayer(const string layerName, LayerActivation activation) {
    WorkContext::updateNetwork(this->networkID);
    WorkContext::updatePlan(WorkContext::curDOPID, true);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;

    Layer<Dtype>* layer;
    bool foundLayer = false;
    for (map<int, void*>::iterator iter = pp->instanceMap.begin();
        iter != pp->instanceMap.end(); iter++) {
        int layerID = iter->first;
        void* instancePtr = iter->second;

        WorkContext::updateLayer(this->networkID, layerID);

        if ((activation == LayerActivation::TrainActivation) && 
            (SLPROP_BASE(activation) == LayerActivation::TestActivation))
            continue;

        if ((activation == LayerActivation::TestActivation) && 
            (SLPROP_BASE(activation) == LayerActivation::TrainActivation))
            continue;

        layer = (Layer<Dtype>*)instancePtr;

        // FIXME: 현재 linear search. 너무 속도가 느리면 개선하자.
        if (SLPROP_BASE(name) == layerName) {
            foundLayer = true;
            layer->layerID = layerID;
            break;
        }
    }

    if (foundLayer)
        return layer;
    else
        return NULL;
}

template <typename Dtype>
Layer<Dtype>* Network<Dtype>::findLayer(const string layerName) {
    return findLayer(layerName, LayerActivation::AllActivation);
}

template <typename Dtype>
vector<Layer<Dtype>*> Network<Dtype>::findLayersByType(int layerType) {
    vector<Layer<Dtype>*> result;

    WorkContext::updateNetwork(this->networkID);
    WorkContext::updatePlan(WorkContext::curDOPID, true);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;

    bool foundLayer = false;
    for (map<int, void*>::iterator iter = pp->instanceMap.begin();
        iter != pp->instanceMap.end(); iter++) {
        int layerID = iter->first;
        void* instancePtr = iter->second;

        WorkContext::updateLayer(this->networkID, layerID);
        Layer<Dtype>* layer = (Layer<Dtype>*)instancePtr;
        layer->layerID = layerID;

        // FIXME: 현재 linear search. 너무 속도가 느리면 개선하자.
        if (WorkContext::curLayerProp->layerType == layerType) {
            result.push_back(layer);
        }
    }

    return result;
}

template<typename Dtype>
Data<Dtype>* Network<Dtype>::findTensor(int nodeID, int devID, string tensorName) {
    WorkContext::updateNetwork(this->networkID);
    WorkContext::updatePlan(WorkContext::curDOPID, true);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;

    // XXX: does not consider multi-device, multi-node situation
    Data<Dtype>* result = (Data<Dtype>*)pp->getTensor(nodeID, devID, tensorName);

    return result;
}

template<typename Dtype>
bool Network<Dtype>::isInnerLayer(int layerID) {
    if (layerID >= SPARAM(SPLITLAYER_START_LAYERID))
        return false;

    return LogicalPlan::isInnerLayer(this->networkID, layerID);
}

template<typename Dtype>
void Network<Dtype>::logNetworkDefString(string networkDef) {
    SASSERT0(this->trainFP != NULL);
    fprintf(this->trainFP, "%s\n", networkDef.c_str());
    fprintf(this->trainFP, "=========================================================\n\n");
    fflush(this->trainFP);
}

template<typename Dtype>
void Network<Dtype>::logTrainHistory(vector<pair<int, string>> params) {
    SASSERT0(this->trainFP != NULL);

    for (int i = 0; i < params.size(); i++) {
        fprintf(this->trainFP, "%d,%s\n", params[i].first, params[i].second.c_str());
    }

    fflush(this->trainFP);
}

template<typename Dtype>
void Network<Dtype>::logMeasureHistory(vector<pair<int, vector<float>>> measures) {
    if (!getMeasureInserted())
        return;
        
    MeasureEntry* measureEntry = MeasureManager::getMeasureEntry(this->networkID); 
    SASSERT0(measureEntry != NULL);

    measureEntry->logHistoryData(measures);
    MeasureManager::releaseMeasureEntry(this->networkID);

}

template<typename Dtype>
void Network<Dtype>::logNetworkDefFile(string networkDefFilePath) {
    std::ifstream file(networkDefFilePath);
    std::string content((std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>());
    logNetworkDefString(content);
}

template<typename Dtype>
void Network<Dtype>::logTrainFile(string content) {
    SASSUME0(this->trainFP != NULL);
    fprintf(this->trainFP, "%s\n", content.c_str());
    fflush(this->trainFP);
}

template class Network<float>;
