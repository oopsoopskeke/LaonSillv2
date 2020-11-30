/**
 * @file ResourceMgmt.h
 * @date 2018-02-08
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef RESOURCEMGMT_H
#define RESOURCEMGMT_H 

#include <map>

typedef enum ResourceItemType_e : int {
    RSC_TYPE_eCPUMEM = 0,
    RSC_TYPE_eGPUMEM,
    RSC_TYPE_eLOCK,
    RES_TYPE_eMax
} ResourceItemType;

typedef struct ResourceItem_s {
    ResourceItemType itemType;
} ResourceItem;

class ResourceMgmt {
public: 
    ResourceMgmt() {}
    virtual ~ResourceMgmt() {}

};

#endif /* RESOURCEMGMT_H */
