/**
 * @file Task.cpp
 * @date 2017-06-14
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "Task.h"
#include "Param.h"
#include "SysLog.h"
#include "ColdLog.h"
#include "MemoryMgmt.h"

using namespace std;

vector<TaskPool*> Task::taskPools;

void Task::allocTaskPool(TaskType taskType) {
    TaskPool *tp = NULL;
    SNEW_ONCE(tp, TaskPool);
    SASSERT0(tp != NULL);
    tp->taskType = taskType;

    int elemCount;
    switch(taskType) {
        case TaskType::AllocTensor:
            elemCount = SPARAM(TASKPOOL_ALLOCTENSOR_ELEM_COUNT);
            break;

        case TaskType::UpdateTensor:
            elemCount = SPARAM(TASKPOOL_UPDATETENSOR_ELEM_COUNT);
            break;

        case TaskType::RunPlan:
            elemCount = SPARAM(TASKPOOL_RUNPLAN_ELEM_COUNT);
            break;

        case TaskType::AllocLayer:
            elemCount = SPARAM(TASKPOOL_ALLOCLAYER_ELEM_COUNT);
            break;

        default:
            SASSERT0(false);
    }

    for (int i = 0; i < elemCount; i++) {
        void               *elem = NULL;
        TaskAllocTensor    *elemTaskAllocTensor = NULL;
        TaskUpdateTensor   *elemTaskUpdateTensor = NULL;
        TaskRunPlan        *elemTaskRunPlan = NULL;
        TaskAllocLayer     *elemTaskAllocLayer = NULL;

        switch (taskType) {
            case TaskType::AllocTensor:
                SNEW_ONCE(elemTaskAllocTensor, TaskAllocTensor);
                SASSUME0(elemTaskAllocTensor != NULL);
                elem = (void*)elemTaskAllocTensor;
                break;
            
            case TaskType::UpdateTensor:
                SNEW_ONCE(elemTaskUpdateTensor, TaskUpdateTensor);
                SASSUME0(elemTaskUpdateTensor != NULL);
                elem = (void*)elemTaskUpdateTensor;
                break;

            case TaskType::RunPlan:
                SNEW_ONCE(elemTaskRunPlan, TaskRunPlan);
                SASSUME0(elemTaskRunPlan != NULL);
                elem = (void*)elemTaskRunPlan;
                break;

            case TaskType::AllocLayer:
                SNEW_ONCE(elemTaskAllocLayer, TaskAllocLayer);
                SASSUME0(elemTaskAllocLayer);
                elem = (void*)elemTaskAllocLayer;
                break;

            default:
                SASSERT0(false);
        }
        SASSERT0(elem != NULL);
       
        TaskBase* tb = (TaskBase*)elem;
        tb->elemID = i;
        tb->taskType = taskType;

        tp->alloc.push_back(elem);
        tp->freeElemIDList.push_back(i);
    }
    taskPools.push_back(tp); 
}

void Task::init() {
    for (int i = 0; i < TaskType::TaskTypeMax; i++) {
        allocTaskPool((TaskType)i);
    }
}

void Task::releaseTaskPool(TaskType taskType) {
    int                 elemCount; 
    TaskAllocTensor    *elemTaskAllocTensor;
    TaskUpdateTensor   *elemTaskUpdateTensor;
    TaskRunPlan        *elemTaskRunPlan;
    TaskAllocLayer     *elemTaskAllocLayer;

    for (int i = 0; i < taskPools[taskType]->alloc.size(); i++) {
        switch(taskType) {
            case TaskType::AllocTensor:
                elemTaskAllocTensor = (TaskAllocTensor*)taskPools[taskType]->alloc[i];
                SDELETE(elemTaskAllocTensor);
                break;

            case TaskType::UpdateTensor:
                elemTaskUpdateTensor = (TaskUpdateTensor*)taskPools[taskType]->alloc[i];
                SDELETE(elemTaskUpdateTensor);
                break;

            case TaskType::RunPlan:
                elemTaskRunPlan = (TaskRunPlan*)taskPools[taskType]->alloc[i];
                SDELETE(elemTaskRunPlan);
                break;

            case TaskType::AllocLayer:
                elemTaskAllocLayer = (TaskAllocLayer*)taskPools[taskType]->alloc[i];
                SDELETE(elemTaskAllocLayer);
                break;

            default:
                SASSERT0(false);
        }
    }
    taskPools[taskType]->alloc.clear();
    SDELETE(taskPools[taskType]);
}

void Task::destroy() {
    for (int i = 0; i < TaskType::TaskTypeMax; i++) {
        releaseTaskPool((TaskType)i);
    }
    taskPools.clear();
}

void* Task::getElem(TaskType taskType) {
    SASSUME0(taskType < TaskType::TaskTypeMax);
    TaskPool *tp = taskPools[taskType];
    unique_lock<mutex> lock(tp->mutex);

    if (tp->freeElemIDList.empty()) {
        lock.unlock();
        COLD_LOG(ColdLog::WARNING, true,
            "there is no free elem for task pool. task pool type=%d", (int)taskType);
        return NULL;
    }

    int elemID = tp->freeElemIDList.front();
    tp->freeElemIDList.pop_front();

    lock.unlock();

    return tp->alloc[elemID];
}

void Task::releaseElem(TaskType taskType, void* elemPtr) {
    SASSUME0(taskType < TaskType::TaskTypeMax);
    TaskPool *tp = taskPools[taskType];

    TaskBase* tb = (TaskBase*)elemPtr;
    int elemID = tb->elemID;

    unique_lock<mutex> lock(tp->mutex);
    tp->freeElemIDList.push_back(elemID);
}
