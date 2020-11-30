/**
 * @file PlanValidator.h
 * @date 2018-03-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef PLANVALIDATIOR_H
#define PLANVALIDATIOR_H 

#include "EnumDef.h"
#include "LayerFunc.h"
#include "LogicalPlan.h"

#include <string>

#include "jsoncpp/json/json.h"

typedef enum PlanValidation_e {
    PLAN_VALIDATION_eSUCCESS = 0,
    PLAN_VALIDATION_eOPEN_NETWORKDEF_FILE_FAILED,
    PLAN_VALIDATION_eJSON_PARSE_FAILED,
    PLAN_VALIDATION_eINVALID_TENSOR_NETWORK_FOR_TRAIN,
    PLAN_VALIDATION_eINVALID_TENSOR_NETWORK_FOR_INFERENCE,
    PLAN_VALIDATION_eNOT_ENOUGH_GPU,
    PLAN_VALIDATION_eMAX
} PlanValidation;

class PlanValidator {
public: 
    PlanValidator() {}
    virtual ~PlanValidator() {}

    static PlanValidation check(std::string jsonString, std::string &networkID,
            int &gpuMBSize);
    static PlanValidation checkFromFile(std::string filePath, std::string &networkID,
            int &gpuMBSize);

private:
    static PlanValidation checkNetworkDef(std::string networkID, bool useCompositeModel, 
            LayerActivation lact, int &gpuMBSize);
    static void cleanup(std::string networkID);
    static PlanValidation walk(std::string networkID, LogicalPlan* lp, LayerActivation lact,
            std::map<std::string, TensorShape>& tensorShapeMap,
            std::map<int, uint64_t>& gpuInOutTensorSizeMap,
            std::map<std::string, uint64_t>& gpuLayerTensorSizeMap);
    static PlanValidation checkInternal(Json::Value rootValue, std::string &networkID,
            int &gpuMBSize);
};

#endif /* PLANVALIDATOR_H */
