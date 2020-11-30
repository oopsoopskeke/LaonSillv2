/**
 * @file LogicalPlan.cpp
 * @date 2017-05-10
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <algorithm>

#include "LogicalPlan.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "BaseLayer.h"
#include "LayerPropList.h"
#include "PropMgmt.h"
#include "WorkContext.h"
#include "MemoryMgmt.h"

using namespace std;

map<string, LogicalPlan*> LogicalPlan::lpMap;
mutex LogicalPlan::lpMapMutex;

void LogicalPlan::cleanup(string networkID) {
    unique_lock<mutex> lock(LogicalPlan::lpMapMutex);
    SASSERT(LogicalPlan::lpMap.find(networkID) != LogicalPlan::lpMap.end(),
        "There is no network ID for the logical plan you are trying to delete."
        " networkID=%s", networkID.c_str());

    LogicalPlan* lp = LogicalPlan::lpMap[networkID];
    LogicalPlan::lpMap.erase(networkID);
    lock.unlock();

    SDELETE(lp);
}

PlanDef* LogicalPlan::findPlanDef(LogicalPlan* lp, int planID) {
    for (int i = 0; i < lp->ppDefs.size(); i++) {
        if (lp->ppDefs[i].planID == planID)
            return &lp->ppDefs[i];
    }

    SASSERT(false, "cannot find plandef for requesting plan ID. planID=%d", planID);
}

int LogicalPlan::getSplitOutputCount(vector<int> inputIDs, vector<int> outputIDs) {
    int count = 0;

    for (int i = 0; i < outputIDs.size(); i++) {
        bool skipCount = false;

        for (int j = 0; j < inputIDs.size(); j++) {
            if (outputIDs[i] == inputIDs[j]) {
                skipCount = true;
                break;
            }
        }

        if (!skipCount)
            count++;
    }

    return count;
}

// XXX: the number of codes for this function is too long!!!!!!! split it
//     build()함수는 아래와 같은 일들을 수행한다.
//  (1) 각 레이어의 정의(PlanDef)를 토대로 해야할 세부 plan들을 생성
//  (2) 각 세부 plan들간의 관계(ex 의존성)를 설정
//  (3) 특수 레이어 케이스(ex. split layer, inplace layer) 처리
//     - inplace layer : 자신의 인풋과 아웃풋이 동일한 경우
//     - split layer : A, B, C 3개의 레이어가 존재하는 경우에..
//                     A의 output이 B,C의 input이 되는 경우를 의미
//
//  planDefMap : key=>layerID, value=>PlanBuildDef
void LogicalPlan::build(string networkID, map<int, PlanBuildDef> planDefMap) {
    // (1) fill input2ID & output2ID map
    map<string, vector<int>> input2IDMap;   // tensor name을 기준으로 input ID map
    map<string, vector<int>> output2IDMap;  // tensor name을 기준으로 output ID map

    for (map<int, PlanBuildDef>::iterator it = planDefMap.begin(); it != planDefMap.end();
        ++it) {
        int key = it->first;
        PlanBuildDef value = it->second;

        for (int i = 0; i < value.inputs.size(); i++) {
            string inputKey = value.inputs[i];
            if (output2IDMap.find(inputKey) == output2IDMap.end()) {
                output2IDMap[inputKey] = {};
            }

            output2IDMap[inputKey].push_back(key);
        }

        for (int i = 0; i < value.outputs.size(); i++) {
            string outputKey = value.outputs[i];
            if (input2IDMap.find(outputKey) == input2IDMap.end()) {
                input2IDMap[outputKey] = {};
            }

            input2IDMap[outputKey].push_back(key);
        }
    }

    // (1-1) sort
    for (map<string, vector<int>>::iterator it = input2IDMap.begin();
        it != input2IDMap.end(); ++it) {
        string key = it->first;
        sort(input2IDMap[key].begin(), input2IDMap[key].end());
    }

    for (map<string, vector<int>>::iterator it=output2IDMap.begin(); it!=output2IDMap.end();
            ++it) {
        string key = it->first;
        sort(output2IDMap[key].begin(), output2IDMap[key].end());
    }

    // (2) fill propDowns
    for (map<int, PlanBuildDef>::iterator it = planDefMap.begin(); it != planDefMap.end();
        ++it) {
        int key = it->first;
        PlanBuildDef value = it->second;
  
        int inputCount = value.inputs.size();
        int propDownCount = value.propDowns.size();
        WorkContext::updateLayer(networkID, value.layerID);

        for (int i = 0; i < inputCount - propDownCount; i++) {
            bool propDownVal = true;
            int propDownIdx = i + propDownCount;
            if (propDownIdx < SLPROP_BASE(defaultPropDown).size()) {
                propDownVal = SLPROP_BASE(defaultPropDown)[propDownIdx];
            }
                
            planDefMap[key].propDowns.push_back(propDownVal);
            SLPROP_BASE(propDown).push_back(propDownVal);
        }
    }

    // (3) generate plans
    LogicalPlan* lp = NULL;
    SNEW(lp, LogicalPlan, networkID);
    SASSUME0(lp != NULL);

    // (3-1) generate forward plans
    for (map<int, PlanBuildDef>::iterator it = planDefMap.begin(); it != planDefMap.end();
        ++it) {
        int key = it->first;

        PlanDef newPlanDef;
        PlanBuildDef planBuildDef = it->second;

        newPlanDef.planID = LP_FORWARD_PLANID(key);
        newPlanDef.planType = PLANTYPE_FORWARD;

        newPlanDef.layerID = key;
        newPlanDef.layerType = planBuildDef.layerType;

        newPlanDef.depCount = 0;
        newPlanDef.notifyList = {};

        for (int i = 0; i < planBuildDef.outputs.size(); i++) {
            string outputName = planBuildDef.outputs[i];

            // 아웃풋이 다른 레이어의 인풋인지 확인한다. 
            // 만약 아웃풋이 다른 레이어의 인풋이 아니라면 output layer이다.
            // 아웃풋 레이어에서는 동일 레이어의 backward plan에게 notify 하면 된다.
            if (output2IDMap.find(outputName) == output2IDMap.end()) {
                newPlanDef.notifyList.push_back(LP_BACKWARD_PLANID(key));
                continue;
            }
            
            // inplace에 대해서 확인
            vector<int> IDList = output2IDMap[outputName];
            bool isInplace = false;
            for (int i = 0; i < IDList.size() - 1; i++) {
                if (key == IDList[i]) {
                    isInplace = true;
                    newPlanDef.notifyList.push_back(LP_FORWARD_PLANID(IDList[i+1]));
                    break;
                }
            }

            if (!isInplace) {
                int nextPlanID = LP_FORWARD_PLANID(output2IDMap[outputName][0]);
                newPlanDef.notifyList.push_back(nextPlanID);
            }

            bool isSplit = ((output2IDMap[outputName].size() > 0) &&
                    (input2IDMap[outputName].size() > 0) &&
                    (input2IDMap[outputName].size() < output2IDMap[outputName].size()) &&
                    (input2IDMap[outputName][input2IDMap[outputName].size() - 1] == 
                     newPlanDef.layerID));

            // 뒤에서 split case에 대해서는 처리하기 때문
            if (isSplit) {
                newPlanDef.notifyList.pop_back();
            }
        }

        lp->ppDefs.push_back(newPlanDef);
    }

    // (3-2) generate backward plans
    for (map<int, PlanBuildDef>::iterator it = planDefMap.begin(); it != planDefMap.end();
        ++it) {
        int key = it->first;

        PlanDef newPlanDef;
        PlanBuildDef planBuildDef = it->second;

        newPlanDef.planID = LP_BACKWARD_PLANID(key);
        newPlanDef.planType = PLANTYPE_BACKWARD;

        newPlanDef.layerID = key;
        newPlanDef.layerType = planBuildDef.layerType;

        newPlanDef.depCount = 0;
        newPlanDef.notifyList = {};

        int propDownCount = 0;

        for (int i = 0; i < planBuildDef.inputs.size(); i++) {
            // learnable layer의 경우에 update plan에게 notify해줘야 한다.
            if (planBuildDef.learnable && planBuildDef.propDowns[i]) {
                newPlanDef.notifyList.push_back(LP_UPDATE_PLANID(key)); 
            }

            if (!planBuildDef.propDowns[i])
                continue;

            string inputName = planBuildDef.inputs[i];

            // 인풋이 다른 레이어의 아웃풋인지 확인한다. 
            // 만약 인풋이 다른 레이어의 아웃풋이 아니라면 딱히 알려줘야할 대상은 없다.
            if (input2IDMap.find(inputName) == input2IDMap.end()) {
                continue;
            }

            // input이 곧 자신인 경우(inplace)에 대해서 확인하고, 그에따른 처리를 한다.
            vector<int> IDList = output2IDMap[inputName];
            bool isInplace = false;
            for (int i = IDList.size() - 1; i > 0; i--) {
                if (key == IDList[i]) {
                    isInplace = true;
                    newPlanDef.notifyList.push_back(LP_BACKWARD_PLANID(IDList[i-1]));
                    break;
                }
            }

            if (!isInplace) {
                int nextPlanID = LP_BACKWARD_PLANID(input2IDMap[inputName][0]);
                newPlanDef.notifyList.push_back(nextPlanID);
            }
        }

        lp->ppDefs.push_back(newPlanDef);
    }

    // (3-3) generate update plans
    for (map<int, PlanBuildDef>::iterator it = planDefMap.begin(); it != planDefMap.end();
        ++it) {
        int key = it->first;

        PlanDef newPlanDef;
        PlanBuildDef planBuildDef = it->second;

        newPlanDef.planID = LP_UPDATE_PLANID(key);
        newPlanDef.planType = PLANTYPE_UPDATE;

        newPlanDef.layerID = key;
        newPlanDef.layerType = planBuildDef.layerType;

        newPlanDef.depCount = 0;

        newPlanDef.notifyList = {};
        int depCount = 0;
        for (int i = 0; i < planBuildDef.inputs.size(); i++) {
            // learnable layer의 경우에 update plan에게 notify해줘야 한다.
            if (planBuildDef.learnable && planBuildDef.propDowns[i]) {
                depCount++;
            }
        }

        if (depCount > 0) {
            lp->ppDefs.push_back(newPlanDef);
        }
    }

    // (3-4) generate split layer
    int curLayerID = SPARAM(SPLITLAYER_START_LAYERID);

    for (map<string, vector<int>>::iterator it = input2IDMap.begin();
        it != input2IDMap.end(); ++it) {
        string key = it->first;
        vector<int> inputIDs = it->second;

        if (output2IDMap.find(key) == output2IDMap.end())
            continue;
        vector<int> outputIDs = output2IDMap[key];

        if (inputIDs.size() == 0 || outputIDs.size() == 0)
            continue;

        if (inputIDs.size() >= outputIDs.size())
            continue;

        SASSERT(inputIDs.size() < outputIDs.size(), "input ID size is not less than "
        		"output ID size of Split layer for %s: %d vs %d",
        		key.c_str(), inputIDs.size(), outputIDs.size());

        // (3-4-0) prepare names for split layer
        int splitOutputCount = getSplitOutputCount(inputIDs, outputIDs);
        vector<string> splitLayerOutputDataNames;
        vector<string> splitLayerInputDataNames;

        int frontSplitLayerID = inputIDs[inputIDs.size() - 1];
        WorkContext::updateLayer(networkID, frontSplitLayerID);

        int splitLayerDataIdx = -1;
        for (int i = 0; i < SLPROP_BASE(output).size(); i++) {
            if (SLPROP_BASE(output)[i] == key) {
                splitLayerDataIdx = i;
                break;
            }
        }
        SASSUME0(splitLayerDataIdx >= 0);

        string splitLayerName = key + "_" + SLPROP_BASE(name) + "_" +
            to_string(splitLayerDataIdx) + "_split";
        char splitLayerTempDataName[64];

        for (int i = 0; i < splitOutputCount; i++) {
            sprintf(splitLayerTempDataName, "%s_%d", splitLayerName.c_str(), i);
            splitLayerOutputDataNames.push_back(string(splitLayerTempDataName));
        }
        splitLayerInputDataNames.push_back(key);

        // (3-4-1) generate split layer's forward plan
        PlanDef newPlanDefForward;
        newPlanDefForward.layerID = curLayerID;
        newPlanDefForward.planID = LP_FORWARD_PLANID(newPlanDefForward.layerID);
        newPlanDefForward.planType = PLANTYPE_FORWARD;
        newPlanDefForward.layerType = (int)Layer<float>::Split;
        newPlanDefForward.depCount = 0;

        newPlanDefForward.notifyList = {};
        for (int i = 0; i < splitOutputCount; i++) {
            int splitOutputID = LP_FORWARD_PLANID(outputIDs[outputIDs.size() - i - 1]);
            PlanDef* splitOutputPlanDef = LogicalPlan::findPlanDef(lp, splitOutputID);
            newPlanDefForward.notifyList.push_back(splitOutputID);
        }
        int splitInputID = LP_FORWARD_PLANID(inputIDs[inputIDs.size() - 1]);
        PlanDef* splitInputPlanDef = LogicalPlan::findPlanDef(lp, splitInputID);

        splitInputPlanDef->notifyList.push_back(newPlanDefForward.planID);

        // (3-4-2) generate split layer's backward plan
        PlanDef newPlanDefBackward;
        newPlanDefBackward.layerID = curLayerID;
        newPlanDefBackward.planID = LP_BACKWARD_PLANID(newPlanDefBackward.layerID);
        newPlanDefBackward.planType = PLANTYPE_BACKWARD;
        newPlanDefBackward.layerType = (int)Layer<float>::Split;

        newPlanDefBackward.depCount = 0;

        for (int i = 0; i < splitOutputCount; i++) {
            int splitOutputID = LP_BACKWARD_PLANID(outputIDs[outputIDs.size() - i - 1]);
            PlanDef* splitOutputPlanDef = LogicalPlan::findPlanDef(lp, splitOutputID);

            splitOutputPlanDef->notifyList.push_back(newPlanDefBackward.planID);
        }

        newPlanDefBackward.notifyList = {};
        splitInputID = LP_BACKWARD_PLANID(inputIDs[inputIDs.size() - 1]);
        splitInputPlanDef = LogicalPlan::findPlanDef(lp, splitInputID);
        newPlanDefBackward.notifyList.push_back(splitInputID);

        lp->ppDefs.push_back(newPlanDefForward);
        lp->ppDefs.push_back(newPlanDefBackward);

        // (3-4-3) create split layer's prop
        LayerProp* newProp = LayerPropList::createLayerProp(networkID, curLayerID,
            "Split");
        LayerPropList::setProp((void*)newProp->prop, "Split", "id", (void*)&curLayerID);
        LayerPropList::setProp((void*)newProp->prop, "Split", "name",
                (void*)&splitLayerName);

        LayerPropList::setProp((void*)newProp->prop, "Split", "input",
            (void*)&splitLayerInputDataNames);
        LayerPropList::setProp((void*)newProp->prop, "Split", "output",
            (void*)&splitLayerOutputDataNames);
        PropMgmt::insertLayerProp(newProp);

        // (3-4-4) change the data names of the layers associated with the split layer
        for (int i = 0; i < splitOutputCount; i++) {
            int splitOutputID = LP_BACKWARD_PLANID(outputIDs[outputIDs.size() - i - 1]);
            PlanDef* splitOutputPlanDef = LogicalPlan::findPlanDef(lp, splitOutputID);
            WorkContext::updateLayer(networkID, splitOutputPlanDef->layerID);

            bool found = false;
            for (int j = 0; j < SLPROP_BASE(input).size(); j++) {
                if (SLPROP_BASE(input)[j] == key) {
                    SLPROP_BASE(input)[j] = splitLayerOutputDataNames[i];
                    found = true;
                    break;
                }
            }
            SASSUME0(found == true);
        }

        curLayerID++;
    }

    // (3-5) fill dep count
    map<int, int> depCountMap;  // key = planID, value = depCount
    for (int i = 0; i < lp->ppDefs.size(); i++) {
        PlanDef planDef = lp->ppDefs[i];
        for (int j = 0; j < planDef.notifyList.size(); j++) {
            int targetPlanID = planDef.notifyList[j];

            if (depCountMap.find(targetPlanID) == depCountMap.end())
                depCountMap[targetPlanID] = 1;
            else
                depCountMap[targetPlanID] += 1;
        }
    }

    for (int i = 0; i < lp->ppDefs.size(); i++) {
        PlanDef *planDef = &lp->ppDefs[i];
        if (depCountMap.find(planDef->planID) != depCountMap.end()) {
            planDef->depCount = depCountMap[planDef->planID];
        } else {
            planDef->depCount = 0;
        }
    }

    // (3-6) 필요없는 플랜 제거
    for (vector<PlanDef>::iterator iter = lp->ppDefs.begin(); iter != lp->ppDefs.end();) {
        PlanDef planDef = (*iter);
        if ((planDef.notifyList.size() == 0) && (planDef.depCount == 0)) {
            iter = lp->ppDefs.erase(iter);
        } else {
            iter++;
        }
    }

    // (3-7) Logical Plan에 등록
    unique_lock<mutex> lock(LogicalPlan::lpMapMutex);
    SASSERT(LogicalPlan::lpMap.find(networkID) == LogicalPlan::lpMap.end(),
        "network ID has been declared redundant. network ID=%s", networkID.c_str());

    LogicalPlan::lpMap[networkID] = lp;
}

void LogicalPlan::printPlanDef(string networkID) {
    unique_lock<mutex> lock(LogicalPlan::lpMapMutex);
    SASSERT(LogicalPlan::lpMap.find(networkID) != LogicalPlan::lpMap.end(),
        "There is no logical plan for the requested network ID. network ID=%s",
        networkID.c_str());

    LogicalPlan* lp = LogicalPlan::lpMap[networkID];
    lock.unlock();

    for (int i = 0; i < lp->ppDefs.size(); i++) {
        char tempBuf[1024];
        int pos = 0;

        if (lp->ppDefs[i].notifyList.size() == 0) {
            strcpy(tempBuf, "None");
        } else {
            for (int j = 0; j < lp->ppDefs[i].notifyList.size(); j++) {
                pos += sprintf(tempBuf + pos, "%d ", lp->ppDefs[i].notifyList[j]);
            }
        }

        STDOUT_BLOCK(cout << "planID : " << lp->ppDefs[i].planID << 
            ", planType : " << lp->ppDefs[i].planType <<
            ", layer ID : " << lp->ppDefs[i].layerID <<
            ", layerType : " << lp->ppDefs[i].layerType <<
            ", depCount : " << lp->ppDefs[i].depCount << 
            " notify List : " << tempBuf << endl;);
    }
}

LogicalPlan* LogicalPlan::getLogicalPlan(string networkID) {
    unique_lock<mutex> lock(LogicalPlan::lpMapMutex);
    SASSERT(LogicalPlan::lpMap.find(networkID) != LogicalPlan::lpMap.end(),
        "There is no logical plan for the requested network ID. network ID=%s",
        networkID.c_str());

    LogicalPlan* lp = LogicalPlan::lpMap[networkID];
    lock.unlock();
    return lp;
}

bool LogicalPlan::isInnerLayer(string networkID, int layerID) {
    LogicalPlan* lp = LogicalPlan::getLogicalPlan(networkID);
    unique_lock<mutex> lock(lp->layerTypeMutex);
    SASSUME0(lp->layerTypeMap.find(layerID) != lp->layerTypeMap.end());
    return lp->layerTypeMap[layerID]; 
}

void LogicalPlan::setLayerType(string networkID, int layerID, bool isInner) {
    LogicalPlan* lp = LogicalPlan::getLogicalPlan(networkID);
    unique_lock<mutex> lock(lp->layerTypeMutex);
    SASSUME0(lp->layerTypeMap.find(layerID) == lp->layerTypeMap.end());
    lp->layerTypeMap[layerID] = isInner;
}
