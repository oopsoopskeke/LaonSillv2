/**
 * @file Task.h
 * @date 2017-06-14
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef TASK_H
#define TASK_H 

#include <list>
#include <mutex>
#include <vector>
#include <string>

#include "Update.h"

typedef enum TaskType_e {
    AllocTensor = 0,
    UpdateTensor,
    RunPlan,
    AllocLayer,
    TaskTypeMax
} TaskType;

typedef struct TaskPool_s {
    TaskType                    taskType;
    std::vector<void*>          alloc;
    std::list<int>              freeElemIDList;
    std::mutex                  mutex;
} TaskPool;

typedef struct TaskBase_s {
    TaskType    taskType;
    int         elemID;
} TaskBase;

typedef enum TaskAllocTensorStep_e {
    Alloc = 0,
    Done
} TaskAllocTensorStep;

typedef struct TaskAllocTensor_s {
    TaskType                        taskType;
    int                             elemID;
    int                             nodeID;
    int                             devID;
    int                             requestThreadID; 
    std::string                     tensorName;
    volatile TaskAllocTensorStep    step;
    volatile void*                  tensorPtr;
} TaskAllocTensor;

typedef struct TaskUpdateTensor_s {
    TaskType                    taskType;
    int                         elemID;
    std::string                 networkID;
    int                         dopID;
    int                         layerID;
    int                         planID;
    std::vector<UpdateParam>    updateParams;
} TaskUpdateTensor;

typedef struct TaskRunPlan_s {
    TaskType    taskType;
    int         elemID;
    int         requestThreadID;
    std::string networkID;
    int         dopID;
    bool        inference;
    bool        useAdhocRun;
    std::string inputLayerName;
    int         channel;
    int         height;
    int         width;
    float*      imageData;
} TaskRunPlan;

typedef struct TaskAllocLayer_s {
    TaskType    taskType;
    int         elemID;
    std::string networkID;
    int         dopID;
    int         layerID;
    int         nodeID;
    int         devID;
    int         requestThreadID;
    int         layerType;
    void*       instancePtr;
} TaskAllocLayer;

class Task {
public: 
    Task() {}
    virtual ~Task() {}

    static void init();
    static void destroy();

    static void* getElem(TaskType taskType);
    static void releaseElem(TaskType taskType, void* elemPtr);

private:
    static void allocTaskPool(TaskType taskType);
    static void releaseTaskPool(TaskType taskType);
    static std::vector<TaskPool*> taskPools;
};
#endif /* TASK_H */
