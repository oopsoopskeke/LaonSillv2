/**
 * @file PlanBuilder.h
 * @date 2017-06-02
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef PLANBUILDER_H
#define PLANBUILDER_H 

#include <string>
#include <vector>
#include <map>

#include "jsoncpp/json/json.h"

#include "Network.h"

class PlanBuilder {
public: 
    PlanBuilder();
    virtual ~PlanBuilder();

    void makeLayer(int layerID, std::string layerName, int layerType,
        std::vector<std::string> input, std::vector<std::string> output);
    void addLayerProp(int layerID, std::string property, std::string value);
    void addNetworkProp(std::string property, std::string value);
    std::string load();

private:
    std::string networkID;
    Network<float>* network;
    std::map<int, std::vector<std::string>> layerConf;
    std::vector<std::string> networkConf;

};
#endif /* PLANBUILDER_H */
