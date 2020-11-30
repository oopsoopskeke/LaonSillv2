/**
 * @file PlanBuilder.cpp
 * @date 2017-06-02
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "PlanBuilder.h"
#include "Network.h"
#include "LayerPropList.h"
#include "PlanParser.h"
#include "MemoryMgmt.h"
#include "SysLog.h"

using namespace std;

PlanBuilder::PlanBuilder() {
    SNEW(this->network, Network<float>);
    SASSUME0(this->network != NULL);
    this->networkID = network->getNetworkID();
}

PlanBuilder::~PlanBuilder() {
    SDELETE(this->network);
}

void PlanBuilder::makeLayer(int layerID, string layerName, int layerType,
    vector<string> input, vector<string> output) {

    SASSERT0(layerConf.find(layerID) == layerConf.end());
    layerConf[layerID] = {};
   
    layerConf[layerID].push_back(string("\"id\" : ") + to_string(layerID));
    layerConf[layerID].push_back(string("\"name\" : \"") + layerName + string("\""));
    layerConf[layerID].push_back(string("\"layer\" : \"") + 
        LayerPropList::getLayerName(layerType) + string("\""));

    if (input.size() > 0) {
        string inputPropString = string("\"input\" : [");
        bool isFirst = true;
        for (int i = 0; i < input.size(); i++) {
            if (isFirst)
                isFirst = false;
            else
                inputPropString += string(",");
            inputPropString += string("\"") + input[i] + string("\"");
        }
        inputPropString += string("]");
        layerConf[layerID].push_back(inputPropString);
    }

    if (output.size() > 0) {
        string outputPropString = string("\"output\" : [");
        bool isFirst = true;
        for (int i = 0; i < output.size(); i++) {
            if (isFirst)
                isFirst = false;
            else
                outputPropString += string(",");
            outputPropString += string("\"") + output[i] + string("\"");
        }
        outputPropString += string("]");
        layerConf[layerID].push_back(outputPropString);
    }
}

void PlanBuilder::addLayerProp(int layerID, string property, string value) {
    SASSERT0(layerConf.find(layerID) != layerConf.end());
    layerConf[layerID].push_back(string("\"") + property + "\" : " + value);
}

void PlanBuilder::addNetworkProp(string property, string value) {
    networkConf.push_back(string("\"") + property + "\" : " + value);
}

string PlanBuilder::load() {
    Json::Value value;
    Json::Reader reader;

    string document;

    SASSERT0(layerConf.size() > 0);
    
    document += string("{ \"layers\" : [ ");

    bool isFirstLayer = true;

    for (map<int, vector<string>>::iterator iter = layerConf.begin(); iter != layerConf.end();
        iter++) {
        int layerID = iter->first;
        vector<string> layerProps = iter->second;

        if (isFirstLayer) {
            isFirstLayer = false;
        } else {
            document += string(", ");
        }

        document += string("{ ");

        for (int i = 0; i < layerProps.size(); i++) {
            if (i > 0) {
                document += string(", ");
            }

            document += layerProps[i];
        }

        document += string("} ");
    }

    document += string("], \"configs\" : { ");
    for (int i = 0; i < networkConf.size(); i++) {
        if (i > 0) {
            document += string(", ");
        }
        document += networkConf[i];
    }

    document += string(" } } ");

    bool parse = reader.parse(document, value);
    SASSERT(parse, "invalid json-format file. error message=%s",
        reader.getFormattedErrorMessages().c_str());

    PlanParser::buildNetwork(this->networkID, value);

    Network<float>* network = Network<float>::getNetworkFromID(this->networkID);
    network->setLoaded();

    return this->networkID;
}
