/**
 * @file PlanParser.h
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef PLANPARSER_H
#define PLANPARSER_H 

#include <string>
#include <vector>
#include <utility>

#include "jsoncpp/json/json.h"

#include "LogicalPlan.h"

class PlanParser {
public: 
    PlanParser() {}
    virtual ~PlanParser() {}

    static std::string loadNetwork(std::string filePath);
    static std::string loadNetworkByJSONString(std::string jsonString, 
            std::string loadPath, int startIterNum);
    static void buildNetwork(std::string networkID, Json::Value val);

    static void setPropValue(Json::Value val, bool isLayer, std::string layerType,
        std::string key, void* prop);
private:
    static std::vector<int64_t> handleInnerLayer(std::string networkID, Json::Value vals,
        std::string parentLayerType, void* parentProp);

    static bool findEnvAndReplace(std::string src, std::string &target);
    static std::string convertEnv(std::string value);
};
#endif /* PLANPARSER_H */
