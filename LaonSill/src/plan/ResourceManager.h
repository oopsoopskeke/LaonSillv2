/**
 * @file ResourceManager.h
 * @date 2017-05-10
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef RESOURCEMANAGER_H
#define RESOURCEMANAGER_H 

#include <vector>

#define NODE_MAX_IP_ADDR_LEN        64
typedef struct GPUDevInfo_t {
    int             nodeID;
    char            nodeIPAddr[NODE_MAX_IP_ADDR_LEN];
    int             nodePortNum; 
    int             devID;
    uint64_t        devMemSize;
} GPUDevInfo;

class ResourceManager {
public: 
    ResourceManager() {}
    virtual ~ResourceManager() {}
    static bool isVaildPlanOption(int option);
    static GPUDevInfo getSingleGPUInfo();

    static void init();
private:
    static std::vector<GPUDevInfo> gpuInfo;
};
#endif /* RESOURCEMANAGER_H */
