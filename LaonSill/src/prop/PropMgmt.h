/**
 * @file PropMgmt.h
 * @date 2017-04-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef PROPMGMT_H
#define PROPMGMT_H 

#include <string>

#include "LayerProp.h"
#include "LayerPropList.h"
#include "NetworkProp.h"
#include "WorkContext.h"

typedef struct LayerPropKey_t {
    std::string     networkID;
    unsigned long   layerID;

    bool operator < (const struct LayerPropKey_t &x) const {
        if (networkID == x.networkID) {
            return layerID < x.layerID;
        } else {
            return networkID < x.networkID;
        }
    }

    LayerPropKey_t(std::string networkID, unsigned long layerID) {
        this->networkID = networkID;
        this->layerID = layerID;
    }
} LayerPropKey;

#define SLPROP(layer, var)                                                                   \
    (((_##layer##PropLayer*)(WorkContext::curLayerProp->prop))->_##var##_)

#define SLPROP_BASE(var)                                                                     \
    (((_BasePropLayer*)(WorkContext::curLayerProp->prop))->_##var##_)

#define SLPROP_LEARN(var)                                                                    \
    (((_LearnablePropLayer*)(WorkContext::curLayerProp->prop))->_##var##_)

#define SNPROP(var)                                                                          \
    (WorkContext::curNetworkProp->_##var##_)


class PropMgmt {
public: 
    PropMgmt() {}
    virtual ~PropMgmt() {}

    static void insertLayerProp(LayerProp* layerProp);
    static void removeLayerProp(std::string networkID);
    static void insertNetworkProp(std::string networkID, _NetworkProp* networkProp);
    static void removeNetworkProp(std::string networkID);

    static LayerProp* getLayerProp(std::string networkID, int layerID);
    static _NetworkProp* getNetworkProp(std::string networkID);
private:
    // FIXME: 맵으로 접근하면 아무래도 느릴 수 밖에 없다. 
    //        쓰레드에서 처리할 job이 변경될때 마다 1번씩 접근하기 때문에 비용이 아주 크지는
    //        않다. 더 좋은 방법이 없을지 고민해보자.
    static std::map<LayerPropKey, LayerProp*> layerPropMap;
    static std::map<std::string, std::vector<int>> net2LayerIDMap;
    static std::map<std::string, _NetworkProp*> networkPropMap;

    static std::mutex layerPropMapMutex;
    static std::mutex net2LayerIDMapMutex;
    static std::mutex networkPropMapMutex;
};

#endif /* PROPMGMT_H */
