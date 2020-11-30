/**
 * @file InputDataProvider.h
 * @date 2017-07-10
 * @author moonhoen lee
 * @brief 
 * @details
 *   현재는 1개의 input layer가 존재한다는 가정아래 설계하였습니다.
 */

#ifndef INPUTDATAPROVIDER_H
#define INPUTDATAPROVIDER_H 

#include <mutex>
#include <vector>
#include <map>
#include <string>

#include "Datum.h"

// Data Reader Type
typedef enum DRType_e {
    DatumType = 0,
    DRTypeMax
} DRType;

typedef void(*CBAllocDRElem)(void** elemPtr);
typedef void(*CBDeallocDRElem)(void* elemPtr);
typedef void(*CBFillDRElem)(void* reader, void* elemPtr);

typedef struct DRCBFuncs_s {
    CBAllocDRElem       allocFunc;
    CBDeallocDRElem     deallocFunc;
    CBFillDRElem        fillFunc;
} DRCBFuncs;

typedef struct InputPool_s {
    volatile int            head;
    int                     tail;
    int                     elemCnt;
    volatile int            remainElemCnt;
    volatile int            activeElemCnt;
    std::mutex              mutex;
    volatile int            waiterTID;
    std::vector<void*>      elemArray;
    void*                   reader;
    DRType                  drType;
} InputPool;

typedef struct InputPoolKey_s {
    std::string networkID;
    int         dopID;
    std::string layerName;

    bool operator < (const struct InputPoolKey_s &x) const {
        if (networkID == x.networkID) {
            if (dopID == x.dopID) {
                return layerName < x.layerName;
            } else {
                return dopID < x.dopID;
            }
        } else {
            return networkID < x.networkID;
        }
    }
} InputPoolKey;

typedef struct PoolInfo_s {
    std::string                 networkID;
    volatile int                threadID;
    volatile int                cleanupThreadID;
    std::vector<InputPoolKey>   inputPoolKeyList;
} PoolInfo;

class InputDataProvider {
public: 
    InputDataProvider() {}
    virtual ~InputDataProvider() {}

    static void init();

    static void addPool(std::string networkID, int dopID, std::string layerName,
            DRType drType, void* reader);
    static void removePool(std::string networkID);

    // for input layer
    static InputPool* getInputPool(std::string networkID, int dopID, std::string layerName);
    static void* getData(InputPool* pool, bool peek);

    // for caller
    static void handleIDP(std::string networkID);

private:
    static std::map<InputPoolKey, InputPool*>   poolMap;
    static std::mutex                           poolMapMutex;
    static std::map<std::string, PoolInfo>      poolInfoMap;
    static std::mutex                           poolInfoMutex;

    static std::map<DRType, DRCBFuncs>          drFuncMap;
    static void handler(std::vector<InputPool*> inputPools);
};

#endif /* INPUTDATAPROVIDER_H */
