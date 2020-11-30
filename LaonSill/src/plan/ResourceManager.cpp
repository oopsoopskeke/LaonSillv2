/**
 * @file ResourceManager.cpp
 * @date 2017-05-10
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "jsoncpp/json/json.h"

#include "ResourceManager.h"
#include "SysLog.h"
#include "Param.h"
#include "PlanOptimizer.h"

using namespace std;

#define CLUSTER_CONF_FILENAME           "cluster.conf"

extern const char*  LAONSILL_HOME_ENVNAME;

vector<GPUDevInfo> ResourceManager::gpuInfo;

void ResourceManager::init() {
    char clusterConfFilePath[PATH_MAX];

    if (strlen(SPARAM(CLUSTER_CONF_PATH)) == 0) {
        sprintf(clusterConfFilePath, "%s/%s", getenv(LAONSILL_HOME_ENVNAME),
            CLUSTER_CONF_FILENAME);
    } else {
        strcpy(clusterConfFilePath, SPARAM(CLUSTER_CONF_PATH));
    }

    filebuf fb;
    if (fb.open(clusterConfFilePath, ios::in) == NULL) {
        SASSERT(false, "cannot open cluster confifuration file. file path=%s",
            clusterConfFilePath);
    }

    Json::Value rootValue;
    istream is(&fb);
    Json::Reader reader;
    bool parse = reader.parse(is, rootValue);

    if (!parse) {
        SASSERT(false, "invalid json-format file. file path=%s. error message=%s",
            clusterConfFilePath, reader.getFormattedErrorMessages().c_str());
    }

    Json::Value nodeInfoList = rootValue["node"];
    for (int i = 0; i < nodeInfoList.size(); i++) {
        Json::Value nodeInfo = nodeInfoList[i];
        SASSERT0(nodeInfo.size() == 5);

        int nodeID = nodeInfo[0].asInt();
        const char* nodeIPAddr = nodeInfo[1].asCString();
        int nodePortNum = nodeInfo[2].asInt();
        int devID = nodeInfo[3].asInt();
        uint64_t devMemSize = nodeInfo[4].asUInt64();

        GPUDevInfo devInfo;
        devInfo.nodeID = nodeID;
        strcpy(devInfo.nodeIPAddr, nodeIPAddr);
        devInfo.nodePortNum = nodePortNum;
        devInfo.devID = devID;
        devInfo.devMemSize = devMemSize;

        gpuInfo.push_back(devInfo);
    }

    fb.close();
}

bool ResourceManager::isVaildPlanOption(int option) {
    if (option == PLAN_OPT_SINGLE_GPU) {
        for (int i = 0; i < gpuInfo.size(); i++) {
            if (gpuInfo[i].nodeID == 0)
                return true;
        }
    } else if (option == PLAN_OPT_MULTI_GPU) {
        int masterDevCount = 0;
        for (int i = 0; i < gpuInfo.size(); i++) {
            if (gpuInfo[i].nodeID == 0)
                masterDevCount++;

            if (masterDevCount > 1)
                return true;
        }
    } else if (option == PLAN_OPT_MULTI_NODE) {
        // TODO: 
        return false;
    } else if (option == PLAN_OPT_VERTICAL_SPLIT) {
        // TODO: 
        return false;
    } else if (option == PLAN_OPT_HORIZONTAL_SPLIT) {
        // TODO: 
        return false;
    }

    return false;
}

GPUDevInfo ResourceManager::getSingleGPUInfo() {
    for (int i = 0; i < ResourceManager::gpuInfo.size(); i++) {
        if (ResourceManager::gpuInfo[i].nodeID != 0)
            continue;

        return ResourceManager::gpuInfo[i];
    }

    SASSERT(false, "cannot find gpu device of master node");

    GPUDevInfo dummy;
    return dummy;
}
