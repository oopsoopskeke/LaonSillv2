/**
 * @file Worker.h
 * @date 2016/10/5
 * @author moonhoen lee
 * @brief 병렬작업을 위한 worker 쓰레드를 관리
 * @details
 * @todo
 */

#ifndef WORKER_H_
#define WORKER_H_

#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <list>
#include <chrono>
#include <string>

#include "common.h"
#include "Job.h"
#include "Update.h"
#include "Task.h"

typedef struct TaskQueue_s {
    std::mutex                  mutex;
    std::list<TaskBase*>        tasks;
} TaskQueue;

/* object detection base network type */
typedef enum WorkerODBaseNetworkType_e : int {
    WORKER_OD_eSSD = 0,
    WORKER_OD_eFRCNN,
    WORKER_OD_eYOLO,
    WORKER_OD_eMAX
} WorkerODBaseNetworkType;

/* image classification basek network type */
typedef enum WorkerICBaseNetworkType_e : int {
    WORKER_IC_eVGG16 = 0,
    WORKER_IC_eInception,
    WORKER_IC_eResNet,
    WORKER_IC_eLeNet,
    WORKER_IC_eDenseNet,
    WORKER_IC_eZFNet,
    WORKER_IC_eMAX
} WorkerICBaseNetworkType;

class Worker {
public:
	                            Worker() {}
	virtual                    ~Worker() {}

	static void                 launchThreads(int taskConsumerCount, 
                                             int jobConsumerCount);
    static void                 joinThreads();
    static int                  pushJob(Job* job);
                                /* called by Sess Thread, Receiver Thread */
	static thread_local int     gpuIdx;

    static int                  getConsumerIdx(int devIdx);

    
    static TaskAllocTensor*     addAllocTensorTask(int consumerIdx, int nodeID, 
                                                  int devID,
                                                  int requestThreadID,
                                                  std::string tensorName);
    static void                 addRunPlanTask(int consumerIdx, std::string networkID,
                                                int dopID, bool inference, 
                                                int requestThreadID, bool useAdhocRun,
                                                std::string inputLayerName, int channel, 
                                                int height, int width, float* imageData);
    static void                 addUpdateTensorTask(int consumerIdx, std::string networkID,
                                                   int dopID, int layerID, int planID,
                                                   std::vector<UpdateParam> updateParams);
    static void                 addAllocLayerTask(int consumerIdx, std::string networkID,
                                                  int dopID, int layerID, int nodeID, 
                                                  int devID, int requestThreadID,
                                                  int layerType, void* instancePtr);

private:
    /**
     * producer에 대한 job control을 위한 변수들
     */
    static std::list<Job*>              jobQueue;
    static std::mutex                   jobQueueMutex;
    static Job*                         popJob();
    static int                          getJobCount();
	static void                         producerThread();

    /**
     * variables and functions for job consumer
     */
	static void                         jobConsumerThread(int consumerIdx);
    static std::list<int>               jcReadyQueue;   /* job consumer ready queue */
    static std::mutex                   jcReadyMutex;
    static void                         insertJCReadyQueue(int consumerIdx);
    static std::vector<int>             getReadyJCs(int count);
    static Job*                         getPubJob(Job* job);

    static bool                         handleJob(Job* job);
    static void                         handleCreateNetworkFromFileJob(Job* job);
    static void                         handleCreateNetwork(Job* job);
    static void                         handleCreateResumeNetwork(Job* job);
    static void                         handleStopNetwork(Job* job);
    static void                         handleDestroyNetwork(Job* job);
    static void                         handleBuildNetwork(Job* job);
    static void                         handleResetNetwork(Job* job);
    static void                         handleRunNetwork(Job*job);
    static void                         handleRunNetworkMiniBatch(Job* job);
    static void                         handleSaveNetwork(Job* job);
    static void                         handleLoadNetwork(Job* job);
    static void                         handleRunNetworkWithInputData(Job* job);
    static void                         handleRunObjectDetectionNetworkWithInput(Job* job);
    static void                         handleRunClassificationNetworkWithInput(Job* job);
    static void                         handleGetMeasureItemName(Job* job);
    static void                         handleGetMeasures(Job* job);
    static void                         handleStartIDP(Job* job);
    static void                         handleGetNetworkEvent(Job* job);
    static void                         handleGetNetworkEventMessage(Job* job);
    static void                         handleCheckNetworkDef(Job* job);
    static void                         handleGetNetworkProgress(Job* job);
    static void                         handleGetNetworkResult(Job* job);


    /**
     * variables and functions for task consumer
     */
	static void                         taskConsumerThread(int consumerIdx,
                                                           int gpuIdx);
    static std::vector<TaskQueue*>      taskQueues;

    static bool                         handleAllocTensorTask(TaskAllocTensor *task);
    static bool                         handleUpdateTensorTask(TaskUpdateTensor* task);
    static bool                         handleRunPlanTask(TaskRunPlan* task);
    static bool                         handleAllocLayerTask(TaskAllocLayer* task);

    /*
     * variables for joining thread
     */
	static std::thread*                 producer;
	static std::vector<std::thread>     consumers;  // for join threads
};

#endif /* WORKER_H_ */
