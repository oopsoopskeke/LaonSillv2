/**
 * @file PlanValidatior.cpp
 * @date 2018-03-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "PlanValidator.h"
#include "Network.h"
#include "MemoryMgmt.h"
#include "PlanParser.h"
#include "LogicalPlan.h"
#include "WorkContext.h"
#include "PropMgmt.h"
#include "common.h"
#include "LayerFunc.h"
#include "ColdLog.h"
#include "MemoryMgmt.h"

#include <map>
#include <vector>
#include <string>

using namespace std;

PlanValidation PlanValidator::walk(string networkID, LogicalPlan* lp, LayerActivation lact, 
        map<string, TensorShape>& tensorShapeMap, 
        map<int, uint64_t>& gpuLayerTensorSizeMap,
        map<string, uint64_t>& gpuInOutTensorSizeMap) {

    vector<PlanDef> planDefs;
    map<int, int> planID2IdxMap;    // key : plan ID, value : planDefs idx
   
    int idx = 0;
    for (int i = 0; i < lp->ppDefs.size(); i++) {
        if (lp->ppDefs[i].planType == PLANTYPE_FORWARD) {
            planDefs.push_back(lp->ppDefs[i]);
            SASSUME0(planID2IdxMap.find(lp->ppDefs[i].planID) == planID2IdxMap.end());
            planID2IdxMap[lp->ppDefs[i].planID] = idx;
            idx++;
        }
    }

    bool *doneIdx;
    SMALLOC(doneIdx, bool, sizeof(bool) * planDefs.size());
    SASSUME0(doneIdx != NULL);

    for (int i = 0; i < planDefs.size(); i++) {
        doneIdx[i] = false;
    }
    int doneCount = 0;

    bool isSuccess = true;
    while (true) {
        bool hasDiff = false;

        if (doneCount == planDefs.size())
            break;

        for (int i = 0; i < planDefs.size(); i++) {
            if (doneIdx[i] == true)
                continue;

            if (planDefs[i].depCount != 0)
                continue;

            doneIdx[i] = true;
            doneCount++;
            hasDiff = true;

            vector<TensorShape> inputShape;
            vector<TensorShape> outputShape;

            WorkContext::updateLayer(networkID, planDefs[i].layerID);

            // 미리 notify 해준다.
            for (int j = 0; j < planDefs[i].notifyList.size(); j++) {
                int targetPlanID = planDefs[i].notifyList[j];
                if (planID2IdxMap.find(targetPlanID) == planID2IdxMap.end())
                    continue;

                int targetIdx = planID2IdxMap[targetPlanID];
                SASSUME0(targetIdx < planDefs.size());
                planDefs[targetIdx].depCount -= 1;
                SASSUME0(planDefs[targetIdx].depCount >= 0);
            }

            if ((lact == LayerActivation::TrainActivation) &&
                (SLPROP_BASE(activation) == LayerActivation::TestActivation)) {
                continue;
            }

            if ((lact == LayerActivation::TestActivation) &&
                (SLPROP_BASE(activation) == LayerActivation::TrainActivation)) {
                continue;
            }

    
            for (int j = 0; j < SLPROP_BASE(input).size(); j++) {
                string inputTensorName = SLPROP_BASE(input)[j];
            
                SASSUME0(tensorShapeMap.find(inputTensorName) != tensorShapeMap.end());
                inputShape.push_back(tensorShapeMap[inputTensorName]);
            }

            bool isValid = LayerFunc::checkShape(planDefs[i].layerType, inputShape, 
                                outputShape);

            if (!isValid) {
                COLD_LOG(ColdLog::ERROR, true,
                    "check shape failed. layer type=%d, id=%d", 
                    planDefs[i].layerType, planDefs[i].layerID);
                isSuccess = false;
                break;
            }

            SASSUME0(SLPROP_BASE(output).size() == outputShape.size());
            for (int j = 0; j < SLPROP_BASE(output).size(); j++) {
                string outputTensorName = SLPROP_BASE(output)[j];
            
                SASSUME0(tensorShapeMap.find(outputTensorName) != tensorShapeMap.end());
                tensorShapeMap[outputTensorName] = outputShape[j];

                SASSUME0(gpuInOutTensorSizeMap.find(outputTensorName) != 
                        gpuInOutTensorSizeMap.end());

                uint64_t outputTensorGPUSize = (uint64_t)(sizeof(float) * outputShape[j].N * 
                        outputShape[j].C * outputShape[j].H * outputShape[j].W);
                gpuInOutTensorSizeMap[outputTensorName] = 
                    max(gpuInOutTensorSizeMap[outputTensorName], outputTensorGPUSize);
            }
            
            uint64_t gpuMemSize = LayerFunc::calcGPUSize(planDefs[i].layerType, inputShape);
            SASSUME0(gpuLayerTensorSizeMap.find(planDefs[i].layerID) != 
                    gpuLayerTensorSizeMap.end());

            gpuLayerTensorSizeMap[planDefs[i].layerID] = 
                max(gpuLayerTensorSizeMap[planDefs[i].layerID], gpuMemSize);
        }

        if ((hasDiff == false) || (isSuccess == false)) {
            isSuccess = false;
            break;
        }
    }

    SFREE(doneIdx);

    if (!isSuccess) {
        if (lact == LayerActivation::TrainActivation)
            return PLAN_VALIDATION_eINVALID_TENSOR_NETWORK_FOR_TRAIN;
        else
            return PLAN_VALIDATION_eINVALID_TENSOR_NETWORK_FOR_INFERENCE;
    }

    return PLAN_VALIDATION_eSUCCESS;
}

PlanValidation PlanValidator::checkNetworkDef(string networkID, bool useCompositeModel,
        LayerActivation lact, int &gpuMBSize) {

    LogicalPlan* lp = LogicalPlan::getLogicalPlan(networkID); 
    SASSERT0(lp != NULL);

    // key : tensor name, value : tensor shape {N, C, H, W}
    map<string, TensorShape> tensorShapeMap; 

    // key : layerID, value : GPU size
    map<int, uint64_t> gpuLayerTensorSizeMap;
    map<string, uint64_t> gpuInOutTensorSizeMap;

    WorkContext::updateNetwork(networkID);

    // tensorShapeMap, gpuLayerTensorSizeMap, gpuInOutTensorSizeMap을 채운다.
    for (int i = 0; i < lp->ppDefs.size(); i++) {
        PlanDef* pDef = &lp->ppDefs[i];
        WorkContext::updateLayer(networkID, pDef->layerID);

        for (int j = 0; j < SLPROP_BASE(output).size(); j++) {
            string outputTensorName = SLPROP_BASE(output)[j];
            if (tensorShapeMap.find(outputTensorName) == tensorShapeMap.end()) {
                tensorShapeMap[outputTensorName] = {0, 0, 0, 0};
                SASSUME0(gpuInOutTensorSizeMap.find(outputTensorName) == 
                        gpuInOutTensorSizeMap.end());
                gpuInOutTensorSizeMap[outputTensorName] = 0UL;
            }
        }

        if (pDef->planType != PLANTYPE_FORWARD)
            continue;

        SASSUME0(gpuLayerTensorSizeMap.find(pDef->layerID) == gpuLayerTensorSizeMap.end());
        gpuLayerTensorSizeMap[pDef->layerID] = 0UL;
    }

    if (useCompositeModel) {
        PlanValidation walkRet = walk(networkID, lp, LayerActivation::TrainActivation,
                tensorShapeMap, gpuLayerTensorSizeMap, gpuInOutTensorSizeMap);
        if (walkRet != PLAN_VALIDATION_eSUCCESS)
            return walkRet;

        walkRet = walk(networkID, lp, LayerActivation::TestActivation, tensorShapeMap,
                gpuLayerTensorSizeMap, gpuInOutTensorSizeMap);
        if (walkRet != PLAN_VALIDATION_eSUCCESS)
            return walkRet;
    } else {
        PlanValidation walkRet = walk(networkID, lp, lact, tensorShapeMap,
                gpuLayerTensorSizeMap, gpuInOutTensorSizeMap);
        if (walkRet != PLAN_VALIDATION_eSUCCESS)
            return walkRet;
    }

    uint64_t totalGPUSize = 0UL;
    map<int, uint64_t>::iterator layerTensorIter;
    map<string, uint64_t>::iterator inOutTensorIter;
    for (layerTensorIter = gpuLayerTensorSizeMap.begin();
            layerTensorIter != gpuLayerTensorSizeMap.end(); ++layerTensorIter) {
        cout << "layer (" << layerTensorIter->first << ") : " << layerTensorIter->second << 
            endl;
        totalGPUSize += ALIGNUP(layerTensorIter->second, SPARAM(CUDA_MEMPAGE_SIZE));
    }

    for (inOutTensorIter = gpuInOutTensorSizeMap.begin();
            inOutTensorIter != gpuInOutTensorSizeMap.end(); ++inOutTensorIter) {
        // 2를 곱해주는 것은 data, grad 때문이다.
        // FIXME: 대부분의 In/Out tensor에서는 data, grad를 모두 가지고 있는 것으로 
        //        알고 있는데 확인해서 그렇지 않다면 수정하자. 
        //        특히 Split Layer를 확인해야 한다!!
        totalGPUSize += ALIGNUP(inOutTensorIter->second, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
        cout << "tensor (" << inOutTensorIter->first << ") : " << inOutTensorIter->second << 
            endl;
    }

    map<string, TensorShape>::iterator shapeIter;
    for (shapeIter = tensorShapeMap.begin(); shapeIter != tensorShapeMap.end();
            ++shapeIter) {
        cout << "tensor (" << shapeIter->first << " ) : ";
        cout << shapeIter->second.N << "x" << shapeIter->second.C << "x" <<
            shapeIter->second.H << "x" << shapeIter->second.W << endl;
    }

    size_t free, total;
    cudaMemGetInfo(&free, &total);

    gpuMBSize = (int)((uint64_t)totalGPUSize / (1024UL * 1024UL));
   
    if (uint64_t(free) < totalGPUSize)
        return PLAN_VALIDATION_eNOT_ENOUGH_GPU;

    return PLAN_VALIDATION_eSUCCESS;
}

void PlanValidator::cleanup(string networkID) {
    Network<float>* network = Network<float>::getNetworkFromID(networkID);

    LogicalPlan::cleanup(networkID);
    PropMgmt::removeNetworkProp(networkID);
    PropMgmt::removeLayerProp(networkID);
    SDELETE(network);
}

PlanValidation PlanValidator::checkInternal(Json::Value rootValue, string &networkID,
        int &gpuMBSize) {
    Network<float>* network = NULL;
    SNEW(network, Network<float>);
    networkID = network->getNetworkID();

    // FIXME: build Network 과정에서 오류가 발생하는 경우에 대한 처리가 필요하다.
    PlanParser::buildNetwork(networkID, rootValue);

    PlanValidation result;
    result = checkNetworkDef(networkID, SNPROP(useCompositeModel),
            LayerActivation::TrainActivation, gpuMBSize);
    if (result != PLAN_VALIDATION_eSUCCESS) {
        cleanup(networkID);
        return result;
    }

    result = checkNetworkDef(networkID, SNPROP(useCompositeModel),
            LayerActivation::TestActivation, gpuMBSize);
    if (result != PLAN_VALIDATION_eSUCCESS) {
        cleanup(networkID);
        return result;
    }

    cleanup(networkID);

    return PLAN_VALIDATION_eSUCCESS;
}

PlanValidation PlanValidator::checkFromFile(string filePath, string &networkID,
        int &gpuMBSize) {
    filebuf fb;
    if (fb.open(filePath.c_str(), ios::in) == NULL) {
        return PLAN_VALIDATION_eOPEN_NETWORKDEF_FILE_FAILED;
    }

    Json::Value rootValue;
    istream is(&fb);
    Json::Reader reader;
    bool parse = reader.parse(is, rootValue);

    fb.close();

    if (!parse) {
        return PLAN_VALIDATION_eJSON_PARSE_FAILED;
    }

    return checkInternal(rootValue, networkID, gpuMBSize);
}

PlanValidation PlanValidator::check(string jsonString, string &networkID, int &gpuMBSize) {
    Json::Value rootValue;
    Json::Reader reader;
    bool parse = reader.parse(jsonString, rootValue);

    if (!parse) {
        return PLAN_VALIDATION_eJSON_PARSE_FAILED;
    }

    return checkInternal(rootValue, networkID, gpuMBSize);
}
