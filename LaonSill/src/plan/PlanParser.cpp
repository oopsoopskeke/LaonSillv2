/**
 * SASSUME0(planInfo != NULL);
 *
 * @file PlanParser.cpp
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include <vector>
#include <string>

#include "PlanParser.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "LayerPropList.h"
#include "Network.h"
#include "ColdLog.h"
#include "MemoryMgmt.h"

using namespace std;

bool PlanParser::findEnvAndReplace(string src, string &target) {
    size_t envStartPos = src.find("$(");
    if (envStartPos == std::string::npos) {
        return true;
    }

    size_t envEndPos = src.find(')');
    SASSERT0(envEndPos != std::string::npos);

    int envLen = envEndPos - envStartPos - 2;
    SASSERT0(envEndPos > envStartPos + 3);
    string envString = src.substr(envStartPos + 2, envLen);

    char* envPath = getenv(envString.c_str());

    COLD_LOG(ColdLog::ERROR, !envPath,
        "environment variable $%s is not set.", envString.c_str());
    SASSERT0(envPath);
    target = "";
    if (envStartPos > 0)
        target += src.substr(0, envStartPos);

    target += string(envPath);

    if (envEndPos < src.size() - 1)
        target += src.substr(envEndPos + 1);

    return false;
}

string PlanParser::convertEnv(string value) {
    string newVal = value;
    string temp;

    while (true) {
        if (findEnvAndReplace(newVal, temp))
            break;

        newVal = temp;
    }

    return newVal;
}

void PlanParser::setPropValue(Json::Value val, bool isLayer, string layerType, string key,
    void* prop) {
    // 파싱에 사용할 임시 변수들
    bool boolValue;
    int64_t int64Value;
    uint64_t uint64Value;
    double doubleValue;
    string stringValue;
    vector<bool> boolArrayValue;
    vector<int64_t> int64ArrayValue;
    vector<uint64_t> uint64ArrayValue;
    vector<double> doubleArrayValue;
    vector<string> stringArrayValue;
    Json::Value arrayValue;

    bool isStructType = false;
    string property;
    string subProperty;
    size_t pos = key.find('.');
    if (pos != string::npos) { 
        property = key.substr(0, pos);
        subProperty = key.substr(pos+1);
        isStructType = true;
        SASSERT0(isLayer);  // layer property만 현재 struct type을 지원한다.
    }

    _NetworkProp* networkProp;
    if (!isLayer) {
        networkProp = (_NetworkProp*)prop;
    }

    switch(val.type()) {
    case Json::booleanValue:
        boolValue = val.asBool();
        if (isLayer && !isStructType) {
            LayerPropList::setProp(prop, layerType.c_str(), key.c_str(), (void*)&boolValue);
        } else if (isLayer && isStructType) {
            LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                subProperty.c_str(), (void*)&boolValue);
        } else {
            NetworkProp::setProp(networkProp, key.c_str(), (void*)&boolValue);
        }
        break;

    case Json::intValue:
        int64Value = val.asInt64();
        if (isLayer && !isStructType) {
            LayerPropList::setProp(prop, layerType.c_str(), key.c_str(), (void*)&int64Value);
        } else if (isLayer && isStructType) {
            LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                subProperty.c_str(), (void*)&int64Value);
        } else {
            NetworkProp::setProp(networkProp, key.c_str(), (void*)&int64Value);
        }
        break;

    case Json::realValue:
        doubleValue = val.asDouble();
        if (isLayer && !isStructType) {
            LayerPropList::setProp(prop, layerType.c_str(), key.c_str(), (void*)&doubleValue);
        } else if (isLayer && isStructType) {
            LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                subProperty.c_str(), (void*)&doubleValue);
        } else {
            NetworkProp::setProp(networkProp, key.c_str(), (void*)&doubleValue);
        }
        break;

    case Json::stringValue:
        stringValue = convertEnv(val.asCString());
        if (isLayer && !isStructType) {
            LayerPropList::setProp(prop, layerType.c_str(), key.c_str(), (void*)&stringValue);
        } else if (isLayer && isStructType) {
            LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                subProperty.c_str(), (void*)&stringValue);
        } else {
            NetworkProp::setProp(networkProp, key.c_str(), (void*)&stringValue);
        }
        break;

    case Json::arrayValue:
        // peek 1st value's type
        SASSERT0(val.size() > 0);
        arrayValue = val[0];
        if (arrayValue.type() == Json::booleanValue) {
            boolArrayValue = {};
            for (int k = 0; k < val.size(); k++) {
                arrayValue = val[k];
                boolArrayValue.push_back(arrayValue.asBool());
            }
            if (isLayer && !isStructType) {
                LayerPropList::setProp(prop, layerType.c_str(), key.c_str(),
                    (void*)&boolArrayValue);
            } else if (isLayer && isStructType) {
                LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                    subProperty.c_str(), (void*)&boolArrayValue);
            } else {
                NetworkProp::setProp(networkProp, key.c_str(), (void*)&boolArrayValue);
            }
        } else if (arrayValue.type() == Json::intValue) {
            int64ArrayValue = {};
            for (int k = 0; k < val.size(); k++) {
                arrayValue = val[k];
                int64ArrayValue.push_back(arrayValue.asInt64());
            }
            if (isLayer && !isStructType) {
                LayerPropList::setProp(prop, layerType.c_str(), key.c_str(),
                    (void*)&int64ArrayValue);
            } else if (isLayer && isStructType) {
                LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                    subProperty.c_str(), (void*)&int64ArrayValue);
            } else {
                NetworkProp::setProp(networkProp, key.c_str(), (void*)&int64ArrayValue);
            }
        } else if (arrayValue.type() == Json::realValue) {
            doubleArrayValue = {};
            for (int k = 0; k < val.size(); k++) {
                arrayValue = val[k];
                doubleArrayValue.push_back(arrayValue.asDouble());
            }
            if (isLayer && !isStructType) {
                LayerPropList::setProp(prop, layerType.c_str(), key.c_str(),
                    (void*)&doubleArrayValue);
            } else if (isLayer && isStructType) {
                LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                    subProperty.c_str(), (void*)&doubleArrayValue);
            } else {
                NetworkProp::setProp(networkProp, key.c_str(), (void*)&doubleArrayValue);
            }
        } else if (arrayValue.type() == Json::stringValue) {
            stringArrayValue = {};
            for (int k = 0; k < val.size(); k++) {
                arrayValue = val[k];
                stringArrayValue.push_back(convertEnv(arrayValue.asString()));
            }
            
            if (isLayer && !isStructType) {
                LayerPropList::setProp(prop, layerType.c_str(), key.c_str(),
                    (void*)&stringArrayValue);
            } else if (isLayer && isStructType) {
                LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                    subProperty.c_str(), (void*)&stringArrayValue);
            } else {
                NetworkProp::setProp(networkProp, key.c_str(), (void*)&stringArrayValue);
            }
        } else {
            SASSERT(false, "Unsupported sub-type for array type. sub_type=%d",
                (int)arrayValue.type());
        }
        break;

    default:
        SASSERT(false, "unsupported json-value. type=%d", val.type());
        break;
    }
}

vector<int64_t> PlanParser::handleInnerLayer(std::string networkID, Json::Value vals,
    string parentLayerType, void* parentProp) {
    vector<int64_t> innerIDList;

    for (int i = 0; i < vals.size(); i++) {
        Json::Value innerLayer = vals[i];

        int layerID = innerLayer["id"].asInt();
        SASSERT(layerID < SPARAM(SPLITLAYER_START_LAYERID),
            "layer ID should less than %d. layer ID=%d",
            SPARAM(SPLITLAYER_START_LAYERID), layerID);
        string layerType = innerLayer["layer"].asCString();

        LayerProp* innerProp = 
            LayerPropList::createLayerProp(networkID, layerID, layerType.c_str());

        vector<string> keys = innerLayer.getMemberNames();

        for (int j = 0; j < keys.size(); j++) {
            string key = keys[j];
            Json::Value val = innerLayer[key.c_str()];

            if (strcmp(key.c_str(), "layer") == 0)
                continue;

            setPropValue(val, true, layerType, key,  (void*)innerProp->prop);
        }

        // new prop를 설정.
        PropMgmt::insertLayerProp(innerProp);

        innerIDList.push_back((int64_t)layerID);
    }

    LayerPropList::setProp(parentProp, parentLayerType.c_str(), "innerLayerIDs",
        (void*)&innerIDList);

    return innerIDList;
}

void PlanParser::buildNetwork(std::string networkID, Json::Value rootValue) {
    // (1) get network property
    _NetworkProp *networkProp = NULL;
    SNEW(networkProp, _NetworkProp);
    SASSUME0(networkProp != NULL);
    Json::Value networkConfDic = rootValue["configs"];

    vector<string> keys = networkConfDic.getMemberNames();
    for (int i = 0; i < keys.size(); i++) {
        string key = keys[i];
        Json::Value val = networkConfDic[key.c_str()];

        setPropValue(val, false, "", key,  (void*)networkProp);
    }
    PropMgmt::insertNetworkProp(networkID, networkProp);

    WorkContext::updateNetwork(networkID);

    // (2) fill layer property
    // logical plan을 만들기 위한 변수들
    map<int, PlanBuildDef> planDefMap;  // key : layerID

    vector<int> innerLayerIDList;
    vector<int> layerIDList;

    Json::Value layerList = rootValue["layers"];
    for (int i = 0; i < layerList.size(); i++) {
        Json::Value layer = layerList[i];
        vector<string> keys = layer.getMemberNames();

        // XXX: 예외처리 해야 한다!!!!
        int layerID = layer["id"].asInt();
        SASSERT(layerID < SPARAM(SPLITLAYER_START_LAYERID),
            "layer ID should less than %d. layer ID=%d",
            SPARAM(SPLITLAYER_START_LAYERID), layerID);

        layerIDList.push_back(layerID);

        string layerType = layer["layer"].asCString();

        LayerProp* newProp = 
            LayerPropList::createLayerProp(networkID, layerID, layerType.c_str());

        // fill prop
        for (int j = 0; j < keys.size(); j++) {
            string key = keys[j];
            Json::Value val = layer[key.c_str()];

            if (strcmp(key.c_str(), "layer") == 0)
                continue;

            if (strcmp(key.c_str(), "innerLayer") == 0) {
                vector<int64_t> innerLayerIDs = 
                    handleInnerLayer(networkID, val, layerType, newProp->prop);

                for (int k = 0; k < innerLayerIDs.size(); k++)
                    innerLayerIDList.push_back((int)innerLayerIDs[k]);
                continue;
            }

            setPropValue(val, true, layerType, key,  (void*)newProp->prop);
        }

        // new prop를 설정.
        PropMgmt::insertLayerProp(newProp);

        // useCompositeModel이 아닌 경우에 필요한 layer만 build할 수 있도록 한다.
        if (!SNPROP(useCompositeModel)) {
            bool doFilter = false;

            _BasePropLayer* basePropLayer = (_BasePropLayer*)(newProp->prop);
            LayerActivation activation = basePropLayer->_activation_;

            if ((SNPROP(status) == NetworkStatus::Train) &&
                (activation == LayerActivation::TestActivation)) {
                doFilter = true;
            }

            if ((SNPROP(status) == NetworkStatus::Test) &&
                (activation == LayerActivation::TrainActivation)) {
                doFilter = true;
            }

            if (doFilter)
                continue;
        }

        // plandef 맵에 추가
        SASSERT(planDefMap.find(layerID) == planDefMap.end(),
            "layer ID has been declared redundant. layer ID=%d", layerID);
        PlanBuildDef newPlanDef;
        newPlanDef.layerID = layerID;
        newPlanDef.layerType = LayerPropList::getLayerType(layerType.c_str());   // TODO:

        vector<string> inputs = LayerPropList::getInputs(layerType.c_str(), newProp->prop);
        vector<string> outputs = LayerPropList::getOutputs(layerType.c_str(), newProp->prop);
        vector<bool> propDowns =
            LayerPropList::getPropDowns(layerType.c_str(), newProp->prop); 

        for (int j = 0; j < inputs.size(); j++) {
            newPlanDef.inputs.push_back(inputs[j]);
        }

        for (int j = 0; j < outputs.size(); j++) {
            newPlanDef.outputs.push_back(outputs[j]);
        }

        for (int j = 0; j < propDowns.size(); j++) {
            newPlanDef.propDowns.push_back(propDowns[j]);
        }

        newPlanDef.isDonator = LayerPropList::isDonator(layerType.c_str(), newProp->prop);
        newPlanDef.isReceiver = LayerPropList::isReceiver(layerType.c_str(), newProp->prop);
        newPlanDef.donatorID = LayerPropList::getDonatorID(layerType.c_str(), newProp->prop);
        newPlanDef.learnable = LayerPropList::isLearnable(layerType.c_str(), newProp->prop);

        planDefMap[layerID] = newPlanDef;
    }

    // (3) Logical Plan을 build 한다.
    LogicalPlan::build(networkID, planDefMap);

    for (int i = 0; i < layerIDList.size(); i++) {
        LogicalPlan::setLayerType(networkID, layerIDList[i], false);
    }

    for (int i = 0; i < innerLayerIDList.size(); i++) {
        LogicalPlan::setLayerType(networkID, innerLayerIDList[i], true);
    }
}

string PlanParser::loadNetwork(string filePath) {
    // (1) 우선 network configuration file 파싱부터 진행
    filebuf fb;
    if (fb.open(filePath.c_str(), ios::in) == NULL) {
        SASSERT(false, "cannot open network-def file. file path=%s", filePath.c_str());
    }

    Json::Value rootValue;
    istream is(&fb);
    Json::Reader reader;
    bool parse = reader.parse(is, rootValue);

    if (!parse) {
        SASSERT(false, "invalid json-format file. file path=%s. error message=%s",
            filePath.c_str(), reader.getFormattedErrorMessages().c_str());
    }
   
    // (2) 파싱에 문제가 없어보이니.. 네트워크 ID 생성
    Network<float>* network = NULL;
    SNEW(network, Network<float>);
    string networkID = network->getNetworkID();

    network->logNetworkDefFile(filePath);

    // (3) 네트워크 빌드
    buildNetwork(networkID, rootValue);

    fb.close();

    network->setLoaded();

    return networkID;
}

string PlanParser::loadNetworkByJSONString(string jsonString, string loadPath,
        int startIterNum) {

    Json::Value rootValue;
    Json::Reader reader;
    bool parse = reader.parse(jsonString, rootValue);

    if (!parse) {
        SASSERT(false, "invalid jsonstring. jsonString=%s. error message=%s",
            jsonString.c_str(), reader.getFormattedErrorMessages().c_str());
    }

    if (loadPath != "")
        rootValue["configs"]["loadPath"] = loadPath;

    if (startIterNum > 0)
        rootValue["configs"]["startIterNum"] = startIterNum + 1;
   
    // (2) 파싱에 문제가 없어보이니.. 네트워크 ID 생성
    Network<float>* network = NULL;
    SNEW(network, Network<float>);
    string networkID = network->getNetworkID();

    Json::StreamWriterBuilder wbuilder;
    wbuilder["indentation"] = "\t";
    string wellFormedJSONString = Json::writeString(wbuilder, rootValue);

    network->logNetworkDefString(wellFormedJSONString);

    // (3) 네트워크 빌드
    buildNetwork(networkID, rootValue);

    network->setLoaded();

    return networkID;
}
