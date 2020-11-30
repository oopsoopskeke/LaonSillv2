/**
 * @file Updater.h
 * @date 2017-06-09
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef UPDATER_H
#define UPDATER_H 

#include <map>
#include <mutex>
#include <string>

#include "Update.h"

typedef struct UpdaterKey_s {
    std::string networkID;
    int layerID;

    bool operator < (const struct UpdaterKey_s &x) const {
        if (networkID == x.networkID) {
            return layerID < x.layerID;
        } else {
            return networkID < x.networkID;
        }
    }
} UpdaterKey;

typedef struct UpdaterValue_s {
    int                 nodeID;
    int                 devID;
    std::vector<void*>  tensorDataPtrs;
    std::vector<void*>  tensorDataHis1Ptrs;
    std::vector<void*>  tensorDataHis2Ptrs;
    bool                reshape;
    bool                access;
    int                 paramCount;
    std::mutex          mutex;
} UpdaterValue;

class Updater {
public: 
    Updater() {}
    virtual ~Updater() {}

    static void addUpdater(std::string networkID, int layerID, int paramCount, int nodeID,
            int devID);

    static void unsetReshape(std::string networkID, int layerId);

    static bool updateParams(std::string networkID, int layerID, int planID, int dopID, 
                            std::vector<UpdateParam> updateParams, bool needSyncGrad);

    static bool updateParams(std::vector<UpdateParam> updateParams);

private:
    static std::map<UpdaterKey, UpdaterValue*>  updaterMap;
    static std::mutex                           updaterMutex;
};

#endif /* UPDATER_H */
