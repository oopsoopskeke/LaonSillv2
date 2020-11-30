/**
 * @file Donator.h
 * @date 2017-02-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef DONATOR_H
#define DONATOR_H 

#include <mutex>
#include <map>

#include "common.h"

typedef struct DonatorData_t {
    uint32_t            donatorID;
    int                 donatorRefCount;
    int                 refCount;
    void*               layerPtr;
    bool                cleanUp;    // true => donator wants to be released
} DonatorData;

template<typename Dtype>
class Donator {
public: 
    Donator() {}
    virtual ~Donator() {}

    static void donate(uint32_t donatorID, void* layerPtr);
    static void receive(uint32_t donatorID, void* layerPtr);
    static void releaseDonator(uint32_t donatorID);
    static void releaseReceiver(uint32_t donatorID); 

private:
    static std::map<uint32_t, DonatorData>  donatorMap;
    static std::mutex                       donatorMutex; 
    // XXX: fix lock related codes to get better performance
};

#endif /* DONATOR_H */
