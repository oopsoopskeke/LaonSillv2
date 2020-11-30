/**
 * @file DebugUtil.h
 * @date 2017-03-23
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef DEBUGUTIL_H
#define DEBUGUTIL_H 

#include <string>

#include "common.h"
#include "Network.h"
#include "BaseLayer.h"

template<typename Dtype>
class DebugUtil {
public: 
    // exclusive...
    enum PrintDataType : int {
        PrintData = 1,
        PrintGrad = 2
    };

    DebugUtil() {}
    virtual ~DebugUtil() {}

    static void printIndent(FILE *fp, int indent);
    static void printData(FILE *fp, Dtype data);
    static void printEdges(FILE *fp, const char* title, Data<Dtype>* data, int flags,
        int indent);
    static void printLayerEdges(FILE *fp, const char* title, Layer<Dtype>* layer, int indent);
    static void printLayerEdgesByLayerID(FILE *fp, const char* title, std::string networkID,
        int layerID, int indent);
    static void printLayerEdgesByLayerName(FILE *fp, const char* title, std::string networkID,
        std::string layerName, int indent);
    static void printNetworkEdges(FILE *fp, const char* title, std::string networkID,
        int indent);
    static void printNetworkParams(FILE *fp, const char* title, std::string networkID,
        int indent);
};
#endif /* DEBUGUTIL_H */
