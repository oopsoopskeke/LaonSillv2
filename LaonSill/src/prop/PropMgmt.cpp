/**
 * @file PropMgmt.cpp
 * @date 2017-04-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "PropMgmt.h"
#include "ColdLog.h"
#include "SysLog.h"
#include "WorkContext.h"
#include "MemoryMgmt.h"

using namespace std;

map<LayerPropKey, LayerProp*> PropMgmt::layerPropMap;
map<std::string, vector<int>> PropMgmt::net2LayerIDMap;
map<std::string, _NetworkProp*> PropMgmt::networkPropMap;

mutex PropMgmt::layerPropMapMutex;
mutex PropMgmt::net2LayerIDMapMutex;
mutex PropMgmt::networkPropMapMutex;

LayerProp* PropMgmt::getLayerProp(string networkID, int layerID) {
    LayerPropKey key(networkID, layerID);

    unique_lock<mutex> layerPropMapLock(PropMgmt::layerPropMapMutex);
    map<LayerPropKey, LayerProp*>::iterator iter = layerPropMap.find(key);

    SASSERT0(iter != layerPropMap.end());

    LayerProp* lp = iter->second;
    layerPropMapLock.unlock();

    SASSERT0(lp->networkID == networkID);
    SASSERT0(lp->layerID == layerID);
    
    return lp;
}

_NetworkProp* PropMgmt::getNetworkProp(string networkID) {
    unique_lock<mutex> networkPropMapLock(PropMgmt::networkPropMapMutex);
    map<string, _NetworkProp*>::iterator iter = networkPropMap.find(networkID);
    SASSERT0(iter != networkPropMap.end());

    _NetworkProp* np = iter->second;

    return np;
}

void PropMgmt::insertLayerProp(LayerProp* layerProp) {
    string networkID = layerProp->networkID;
    int layerID = layerProp->layerID;
    unique_lock<mutex> net2LayerIDMapLock(PropMgmt::net2LayerIDMapMutex);
    map<string, vector<int>>::iterator layerIDIter = net2LayerIDMap.find(networkID);
    if (layerIDIter == net2LayerIDMap.end())
        net2LayerIDMap[networkID] = {};
    net2LayerIDMap[networkID].push_back(layerID);
    net2LayerIDMapLock.unlock();

    LayerPropKey key(networkID, layerID);
    unique_lock<mutex> layerPropMapLock(PropMgmt::layerPropMapMutex);
    map<LayerPropKey, LayerProp*>::iterator iter = layerPropMap.find(key);
    SASSERT(iter == layerPropMap.end(), "Layer prop for network %s and layer ID %d"
    		" already inserted.", networkID.c_str(), layerID);

    layerPropMap[key] = layerProp;
}

void PropMgmt::removeLayerProp(string networkID) {
    // XXX: 메모리 잘 해제되는지 확인해야 한다.
    unique_lock<mutex> net2LayerIDMapLock(PropMgmt::net2LayerIDMapMutex);
    map<string, vector<int>>::iterator iter = net2LayerIDMap.find(networkID);
    if (iter == net2LayerIDMap.end()) {
        COLD_LOG(ColdLog::WARNING, true,
            "specific networkID is not registered yet. network ID=%s", networkID.c_str());
    }

    vector<int> layerIDList = iter->second;
    net2LayerIDMapLock.unlock();

    vector<int>::iterator layerIDIter;
    for (layerIDIter = layerIDList.begin(); layerIDIter != layerIDList.end(); ) {
        int layerID = (*layerIDIter);
        LayerProp* lpp = getLayerProp(networkID, layerID);
        LayerPropKey key(networkID, layerID);
        unique_lock<mutex> layerPropMapLock(PropMgmt::layerPropMapMutex);
        layerPropMap.erase(key);
        layerPropMapLock.unlock();
        SDELETE(lpp);

        layerIDIter = layerIDList.erase(layerIDIter);
    }

    net2LayerIDMapLock.lock();
    net2LayerIDMap.erase(networkID);
    net2LayerIDMapLock.unlock();
}

void PropMgmt::insertNetworkProp(string networkID, _NetworkProp* networkProp) {
    unique_lock<mutex> networkPropMapLock(PropMgmt::networkPropMapMutex);
    map<string, _NetworkProp*>::iterator iter = networkPropMap.find(networkID);
    SASSERT0(iter == networkPropMap.end());

    networkPropMap[networkID] = networkProp;
}

void PropMgmt::removeNetworkProp(string networkID) {
    _NetworkProp* networkProp = getNetworkProp(networkID);
    networkPropMap.erase(networkID);
    SDELETE(networkProp);
}
