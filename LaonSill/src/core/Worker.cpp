/*
 * Worker.cpp
 *
 *  Created on: 2016. 10. 5.
 *      Author: moonhoen lee
 */

#include <math.h>

#include "Cuda.h"
#include "Worker.h"

#include "Debug.h"
#include "Network.h"
#include "Param.h"
#include "ColdLog.h"
#include "HotLog.h"
#include "SysLog.h"
#include "Broker.h"
#include "ThreadMgmt.h"
#include "Updater.h"
#include "WorkContext.h"
#include "PlanParser.h"
#include "LayerFunc.h"
#include "PropMgmt.h"
#include "StdOutLog.h"

#include "InputDataProvider.h"
#include "AnnotationDataLayer.h"
#include "DetectionOutputLayer.h"
#include "RoITestLiveInputLayer.h"
#include "FrcnnTestLiveOutputLayer.h"
#include "LiveDataInputLayer.h"

#include "MeasureEntry.h"
#include "MeasureManager.h"
#include "MemoryMgmt.h"
#include "DebugUtil.h"
#include "ImageUtil.h"

#include "frcnn_common.h"   // for use nsm() func
#include "YOLOLossLayer.h"
#include "NetworkRecorder.h"
#include "PlanValidator.h"
#include "MeasureLayer.h"

using namespace std;

thread_local int        Worker::gpuIdx;

list<Job*>              Worker::jobQueue;
mutex                   Worker::jobQueueMutex;

list<int>               Worker::jcReadyQueue;   /* job consumer ready queue */
mutex                   Worker::jcReadyMutex;

vector<TaskQueue*>      Worker::taskQueues;

thread*                 Worker::producer;
vector<thread>          Worker::consumers;

void Worker::producerThread() {
    int threadID = ThreadMgmt::getThreadID(ThreadType::Producer, 0);
    ThreadMgmt::setThreadReady(threadID);
    COLD_LOG(ColdLog::INFO, true, "producer thread starts");
    
    HotLog::initForThread();

    // (1) 서버가 준비 될때까지 대기 한다.
    while (!ThreadMgmt::isReady()) {
        sleep(0);
    }

    // (2) 메인 루프
    while (true) {
        ThreadEvent event =
            ThreadMgmt::wait(threadID, SPARAM(PRODUCER_PERIODIC_CHECK_TIME_MS)); 

        int wakeupCount = Worker::getJobCount();
        vector<int> jcIDs = Worker::getReadyJCs(wakeupCount);

        for (int i = 0; i < jcIDs.size(); i++) {
            int targetThreadID = ThreadMgmt::getThreadID(ThreadType::JobConsumer, jcIDs[i]);
            ThreadMgmt::signal(targetThreadID, ThreadEvent::Wakeup);
        }

        if (event == ThreadEvent::Halt)
            break;
    }

    COLD_LOG(ColdLog::INFO, true, "producer thread ends");
    HotLog::markExit();
}

bool Worker::handleAllocTensorTask(TaskAllocTensor* task) {
    if (task->nodeID != SPARAM(NODE_ID)) {
        // alloc tensor to other nodes.
        SASSERT0(false);        // not implemented yet
    }

    // XXX: float형 코딩으로 박지 말고, 설정에 따라서 template date type을 설정하도록 수정해야
    //     한다. 
    if (task->step == TaskAllocTensorStep::Alloc) {
        Data<float>* tensor = NULL;
        SNEW(tensor, Data<float>, task->tensorName);
        SASSERT0(tensor != NULL);

        task->tensorPtr = tensor;
        task->step = TaskAllocTensorStep::Done;
        
        ThreadMgmt::signal(task->requestThreadID, ThreadEvent::Wakeup);
    }

    return true;
}

bool Worker::handleUpdateTensorTask(TaskUpdateTensor* task) {
    bool ret = Updater::updateParams(task->networkID, task->layerID, task->planID,
        task->dopID, task->updateParams, false);
    return ret;
}

bool Worker::handleRunPlanTask(TaskRunPlan* task) {
    WorkContext::updateNetwork(task->networkID);
    WorkContext::updatePlan(task->dopID, true);

    PhysicalPlan* pp = PhysicalPlan::getCurPhysicalPlan();
    PlanInfo* planInfo = WorkContext::curPlanInfo;

    // (1) 복원해야 할 정보들을 기입하고, useAdhocRun가 True이면 임시 inference 정보를
    //    기입한다. 
    int recoverEpochIdx;
    int recoverMiniBatchIdx;
    int recoverEpochCount;
    int recoverMiniBatchCount;
    int recoverIterationCount;
    int recoverDoneCount;
    NetworkStatus recoverNetworkStatus;

    if (task->useAdhocRun) {
        SASSUME0(task->inference);
        recoverEpochIdx = planInfo->curEpochIndex;
        recoverMiniBatchIdx = planInfo->curMiniBatchIndex;
        recoverEpochCount = planInfo->epochCount;
        recoverMiniBatchCount = planInfo->miniBatchCount;
        recoverDoneCount = planInfo->doneCount;
        recoverIterationCount = SNPROP(iterations);
        recoverNetworkStatus = SNPROP(status);

        planInfo->curEpochIndex = 0;
        planInfo->curMiniBatchIndex = 0;
        planInfo->epochCount = 1;
        planInfo->miniBatchCount = 1;   // FIXME: 한번 이상의 inference를 하는 경우에는 해당 
                                        //        값이 벼경이 되어야 할 것으로 보인다. 추후에
                                        //        수정 필요.
        planInfo->doneCount = 0;
        SNPROP(iterations) = 0;
        SNPROP(status) = NetworkStatus::Test;

        InputLayer<float>* commonInputLayer;
        Network<float>* network;
        network = Network<float>::getNetworkFromID(task->networkID);
        commonInputLayer = (InputLayer<float>*)network->findLayer(task->inputLayerName,
            LayerActivation::TestActivation);
        SASSUME0(commonInputLayer != NULL); 
        WorkContext::updateLayer(task->networkID, commonInputLayer->layerID);
        commonInputLayer->feedImage(task->channel, task->height, task->width,
                task->imageData);
    }

    // (2) plan을 실행한다.
    bool canRunPlan = true;
    while (canRunPlan) {
        canRunPlan = pp->runPlan(task->inference);
    }

    // (3) 복원해야 할 정보가 있으면 복원한다.
    if (task->useAdhocRun) {
        planInfo->curEpochIndex = recoverEpochIdx;
        planInfo->curMiniBatchIndex = recoverMiniBatchIdx;
        planInfo->epochCount = recoverEpochCount;
        planInfo->miniBatchCount = recoverMiniBatchCount;
        SNPROP(iterations) = recoverIterationCount;
        SNPROP(status) = recoverNetworkStatus;
    }

    bool jobRemain = pp->generatePlan(true, task->useAdhocRun);

    if (jobRemain) {
        return false;
    } else {
        bool runFinished = false;
        unique_lock<mutex> lock(WorkContext::curPlanInfo->planMutex);
        WorkContext::curPlanInfo->doneCount += 1;
        if (WorkContext::curPlanInfo->doneCount == WorkContext::curPlanInfo->dopCount) {
            runFinished = true;
            // XXX: multi GPU 상황에서(즉, dopCount > 1인 상황에서) 제대로 동작하는지
            //     검증필요. 그전에 multi GPU를 제대로 구현해야 함.
            if (task->useAdhocRun) {
                planInfo->doneCount = recoverDoneCount;
            }
        }
        lock.unlock();

        if (runFinished) {
            ThreadMgmt::signal(task->requestThreadID, ThreadEvent::Wakeup); 
        }
        
        return true;
    }
}

bool Worker::handleAllocLayerTask(TaskAllocLayer* task) {
    if (task->nodeID != SPARAM(NODE_ID)) {
        // alloc tensor to other nodes.
        SASSERT0(false);        // not implemented yet
    }

    WorkContext::updateLayer(task->networkID, task->layerID);
    WorkContext::updatePlan(task->dopID, true);

    SASSERT0(LayerFunc::allocLayerTensors(task->layerType, task->instancePtr) == true);
    ThreadMgmt::signal(task->requestThreadID, ThreadEvent::Wakeup);

    return true;
}

void Worker::taskConsumerThread(int consumerIdx, int gpuIdx) {
    int threadID = ThreadMgmt::getThreadID(ThreadType::TaskConsumer, consumerIdx);
    ThreadMgmt::setThreadReady(threadID);
    bool doLoop = true;
	Worker::gpuIdx = gpuIdx;

    HotLog::initForThread();

    COLD_LOG(ColdLog::INFO, true, "task consumer thread #%d (GPU:#%d) starts", consumerIdx,
        gpuIdx);

    while (!ThreadMgmt::isReady()) {
        sleep(0);
    }


	// 리소스 초기화
	checkCudaErrors(cudaSetDevice(gpuIdx));
	checkCUBLAS(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

    vector<TaskBase*> remainTasks;
    while (doLoop) {
        ThreadEvent event =
            ThreadMgmt::wait(threadID, SPARAM(TASK_CONSUMER_PERIODIC_CHECK_TIME_MS)); 

        if (event == ThreadEvent::Wakeup || event == ThreadEvent::Timeout) {
            vector<TaskBase*> tasks;

            TaskQueue *tq = taskQueues[consumerIdx];
            unique_lock<mutex> lock(tq->mutex);

            while (!tq->tasks.empty()) {
                tasks.push_back(tq->tasks.front());
                tq->tasks.pop_front();
            }
            lock.unlock();

            for (int i = 0; i < remainTasks.size(); i++) {
                tasks.push_back(remainTasks[i]);
            }
            remainTasks.clear();

            bool hasRemainTask = false;
            for (int i = 0; i < tasks.size(); i++) {
                bool taskDone;
                switch (tasks[i]->taskType) {
                    case TaskType::AllocTensor:
                        taskDone = handleAllocTensorTask((TaskAllocTensor*)tasks[i]);
                        break;

                    case TaskType::UpdateTensor:
                        taskDone = handleUpdateTensorTask((TaskUpdateTensor*)tasks[i]);
                        break;

                    case TaskType::RunPlan:
                        taskDone = handleRunPlanTask((TaskRunPlan*)tasks[i]);
                        break;

                    case TaskType::AllocLayer:
                        taskDone = handleAllocLayerTask((TaskAllocLayer*)tasks[i]);
                        break;

                    default:
                        SASSUME0(false);
                }

                if (!taskDone) {
                    remainTasks.push_back(tasks[i]);
                    hasRemainTask = true;
                } else if (tasks[i]->taskType != TaskType::AllocTensor) {
                    // Alloc Tensor의 경우에는 caller가 release한다.
                    Task::releaseElem(tasks[i]->taskType, (void*)tasks[i]);
                }
            }

            // 남은 task가 있다면 자기 스스로를 깨운다.
            if (hasRemainTask)
                ThreadMgmt::signal(threadID, ThreadEvent::Wakeup);
        }

        if (event == ThreadEvent::Halt)
            break;
    }

	// 리소스 정리
	checkCUBLAS(cublasDestroy(Cuda::cublasHandle));
	checkCUDNN(cudnnDestroy(Cuda::cudnnHandle));

    HotLog::markExit();
    COLD_LOG(ColdLog::INFO, true, "task consumer thread #%d (GPU:#%d) ends", consumerIdx,
        gpuIdx);
}

void Worker::jobConsumerThread(int consumerIdx) {
    int threadID = ThreadMgmt::getThreadID(ThreadType::JobConsumer, consumerIdx);
    ThreadMgmt::setThreadReady(threadID);
    bool doLoop = true;

    HotLog::initForThread();

    COLD_LOG(ColdLog::INFO, true, "job consumer thread #%d (GPU:#%d) starts", consumerIdx,
        gpuIdx);

    while (!ThreadMgmt::isReady()) {
        sleep(0);
    }

    while (doLoop) {
        ThreadEvent event = ThreadMgmt::wait(threadID, 0UL);

        if (event == ThreadEvent::Halt) {
            break;
        }

        Job* job = Worker::popJob();
        if (job == NULL)
            continue;

        doLoop = handleJob(job);

        SDELETE(job);

        Worker::insertJCReadyQueue(consumerIdx);
    }

    HotLog::markExit();
    COLD_LOG(ColdLog::INFO, true, "job consumer thread #%d (GPU:#%d) ends", consumerIdx,
        gpuIdx);
}

Job* Worker::getPubJob(Job* job) {
    SASSUME0(job->hasPubJob());
    unique_lock<mutex> reqPubJobMapLock(Job::reqPubJobMapMutex); 
    Job *pubJob = Job::reqPubJobMap[job->getJobID()];
    SASSUME0(pubJob != NULL);
    Job::reqPubJobMap.erase(job->getJobID());
    reqPubJobMapLock.unlock();
    SASSUME0(pubJob->getType() == job->getPubJobType());
    return pubJob;
}

void Worker::handleCreateNetworkFromFileJob(Job* job) {
    string networkID = PlanParser::loadNetwork(job->getStringValue(0));
    
    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleCreateNetwork(Job* job) {
    string networkID = PlanParser::loadNetworkByJSONString(job->getStringValue(0), "", 0); 

    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleCreateResumeNetwork(Job* job) {
    string networkID = job->getStringValue(0);
    int keepHistoryValue = job->getIntValue(1);

    string newNetworkID = Network<float>::createResumeNetwork(networkID, "",
            (bool)keepHistoryValue);

    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::StringType, strlen(newNetworkID.c_str()),
            (void*)newNetworkID.c_str());

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleStopNetwork(Job* job) {
    string networkID = job->getStringValue(0);
    Network<float>::stopNetwork(networkID);

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleDestroyNetwork(Job* job) {
    string networkID = job->getStringValue(0);

    // XXX: 네트워크가 제거될 수 있는 상황인지에 대한 파악을 해야하고, 그것에 따른 에러처리가
    //      필요하다.
    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    SASSERT0(network->getLoaded());

    if (SPARAM(USE_INPUT_DATA_PROVIDER)) {
        InputDataProvider::removePool(networkID);
    }

    LogicalPlan::cleanup(networkID);

    if (network->getBuilt())
        PhysicalPlan::removePlan(networkID);

    PropMgmt::removeNetworkProp(networkID);
    PropMgmt::removeLayerProp(networkID);
    SDELETE(network);

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleBuildNetwork(Job* job) {
    string networkID = job->getStringValue(0);
    int epochs = job->getIntValue(1);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->build(epochs);

    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleResetNetwork(Job* job) {
    string networkID = job->getStringValue(0);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->reset();

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleRunNetwork(Job* job) {
    string networkID = job->getStringValue(0);
    int inference = job->getIntValue(1);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);

    if (SPARAM(USE_INPUT_DATA_PROVIDER)) {
        WorkContext::updateNetwork(networkID);

        Job* startIDPJob = NULL;
        SNEW(startIDPJob, Job, JobType::StartInputDataProvider);   // InputDataProvider
        SASSUME0(startIDPJob != NULL);
        startIDPJob->addJobElem(Job::StringType, strlen(networkID.c_str()),
            (void*)networkID.c_str());
        Worker::pushJob(startIDPJob);
    }

    network->run((bool)inference);

    ThreadMgmt::wait(WorkContext::curThreadID, 0);

    int clientError = 1;
    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::IntType, 1, (void*)&clientError);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleRunNetworkMiniBatch(Job* job) {
    string networkID = job->getStringValue(0);
    int inference = job->getIntValue(1);
    int miniBatchIdx = job->getIntValue(2);

    int clientError = 1;

    if (SPARAM(USE_INPUT_DATA_PROVIDER)) {
        COLD_LOG(ColdLog::ERROR, true, "run network minibatch is not supported in IDP mode");
        clientError = 0;
    } else {
        Network<float>* network = Network<float>::getNetworkFromID(networkID);
        network->runMiniBatch((bool)inference, miniBatchIdx);

        ThreadMgmt::wait(WorkContext::curThreadID, 0);
    }

    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::IntType, 1, (void*)&clientError);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleSaveNetwork(Job* job) {
    string networkID = job->getStringValue(0);
    string filePath = job->getStringValue(1);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->save(filePath);

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleLoadNetwork(Job* job) {
    string networkID = job->getStringValue(0);
    string filePath = job->getStringValue(1);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->load(filePath);

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleGetMeasureItemName(Job* job) {
    string networkID = job->getStringValue(0);

    MeasureEntry* entry = MeasureManager::getMeasureEntry(networkID);
    int itemCount;
    Job* pubJob = getPubJob(job);

    if (entry == NULL) {
        itemCount = -1;
        pubJob->addJobElem(Job::IntType, 1, (void*)&itemCount);
    } else {
        vector<string> itemNames = entry->getItemNames();
        MeasureManager::releaseMeasureEntry(networkID);

        itemCount = itemNames.size();
        pubJob->addJobElem(Job::IntType, 1, (void*)&itemCount);

        for (int i = 0; i < itemCount; i++) {
            pubJob->addJobElem(Job::StringType, strlen(itemNames[i].c_str()),
                (void*)itemNames[i].c_str());
        }
    }

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleGetMeasures(Job* job) {
    string networkID = job->getStringValue(0);
    int isForward = job->getIntValue(1);
    int start = job->getIntValue(2);
    int count = job->getIntValue(3);

    int startIterNum;
    int measureCount;

    MeasureEntry* entry = MeasureManager::getMeasureEntry(networkID);
    Job* pubJob = getPubJob(job);

    if (entry == NULL) {
        int N = -1;
        pubJob->addJobElem(Job::IntType, 1, (void*)&N);
        pubJob->addJobElem(Job::IntType, 1, (void*)&N);
        pubJob->addJobElem(Job::IntType, 1, (void*)&N);
        pubJob->addJobElem(Job::IntType, 1, (void*)&N);
    } else {
        WorkContext::updateNetwork(networkID);

        int itemCount = entry->getItemNames().size();
        float* data = NULL;
        int allocSize = sizeof(float) * count * itemCount;
        SMALLOC(data, float, allocSize);
        SASSUME0(data != NULL);

        int realStart = start;
        int realCount = count;
        SASSUME0(SNPROP(startIterNum) >= 0);

        if (isForward) {
            realStart = max(0, start - (int)SNPROP(startIterNum));
            realCount = max(0, count - (start - realStart));
        }

        entry->getData(realStart, realCount, (bool)isForward, &startIterNum, &measureCount,
                data); 
        MeasureManager::releaseMeasureEntry(networkID);

        startIterNum += SNPROP(startIterNum);

        int N = measureCount * itemCount;
        pubJob->addJobElem(Job::IntType, 1, (void*)&N);
        pubJob->addJobElem(Job::IntType, 1, (void*)&startIterNum);

        int totalIterCount;
        int curIterCount;

        WorkContext::getNetworkProgress(networkID, curIterCount, totalIterCount);
        pubJob->addJobElem(Job::IntType, 1, (void*)&curIterCount);
        pubJob->addJobElem(Job::IntType, 1, (void*)&totalIterCount);

        if (N > 0)
            pubJob->addJobElem(Job::FloatArrayType, N, data);

        SFREE(data);
    }

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleGetNetworkEvent(Job* job) {
    string networkID = job->getStringValue(0);

    vector<NetworkEvent> events;
    NetworkRecorder::getEvents(networkID, events);

    Job* pubJob = getPubJob(job);
    int eventCount = events.size();
    pubJob->addJobElem(Job::IntType, 1, (void*)&eventCount);

    for (int i = 0; i < events.size(); i++) {
        int eventType = (int)events[i].eventType;
        pubJob->addJobElem(Job::IntType, 1, (void*)&eventType);

        string eventTimeString = NetworkRecorder::timeToString(events[i].eventTime);
        pubJob->addJobElem(Job::StringType, strlen(eventTimeString.c_str()),
                (void*)eventTimeString.c_str());

        int layerID = events[i].layerID;
        pubJob->addJobElem(Job::IntType, 1, (void*)&layerID);

        string msg = events[i].msg;
        pubJob->addJobElem(Job::StringType, strlen(msg.c_str()), (void*)msg.c_str());
    }

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleGetNetworkEventMessage(Job* job) {
    string networkID = job->getStringValue(0);

    vector<string> msgs;
    NetworkRecorder::getEventMsgs(networkID, msgs);

    Job* pubJob = getPubJob(job);
    int eventCount = msgs.size();
    pubJob->addJobElem(Job::IntType, 1, (void*)&eventCount);

    for (int i = 0; i < msgs.size(); i++) {
        string eventMsg = msgs[i];
        pubJob->addJobElem(Job::StringType, strlen(eventMsg.c_str()),
                (void*)eventMsg.c_str());
    }

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleRunNetworkWithInputData(Job* job) {
    string networkID = job->getStringValue(0);
    int channel = job->getIntValue(1);
    int height = job->getIntValue(2);
    int width = job->getIntValue(3);
    int coordRelative = job->getIntValue(4);
    float* imageData = job->getFloatArray(5);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);

    std::vector<Layer<float>*> inputLayers =
        network->findLayersByType(Layer<float>::AnnotationData);
    SASSUME0(inputLayers.size() == 1);
    AnnotationDataLayer<float>* inputLayer = (AnnotationDataLayer<float>*)inputLayers[0];

    std::vector<Layer<float>*> outputLayers =
        network->findLayersByType(Layer<float>::DetectionOutput);
    SASSUME0(outputLayers.size() == 1);
    DetectionOutputLayer<float>* outputLayer =
        (DetectionOutputLayer<float>*)outputLayers[0];

    inputLayer->feedImage(channel, height, width, imageData);

    network->runMiniBatch(true, 0);
    ThreadMgmt::wait(WorkContext::curThreadID, 0);

    Job* pubJob = getPubJob(job);

    int count = outputLayer->_outputData[0]->getCount();
    const float* result = outputLayer->_outputData[0]->host_data();
    int resultCount = 0;

    for (int i = 0; i < count; i += 7) {
    	if (result[i + 1] == 15) {
    		resultCount++;
    	}
    }
    pubJob->addJobElem(Job::IntType, 1, (void*)&resultCount);

    float left, top, right, bottom;
    for (int i = 0; i < count; i += 7) {
    	if (result[i + 1] != 15) {
    		continue;
    	}

    	left	= std::min(std::max(result[i + 3], 0.f), 1.f);
    	top		= std::min(std::max(result[i + 4], 0.f), 1.f);
    	right	= std::min(std::max(result[i + 5], 0.f), 1.f);
    	bottom	= std::min(std::max(result[i + 6], 0.f), 1.f);

    	if (coordRelative == 0) {
    		left    = int(left * width);
			top     = int(top * height);
			right   = int(right * width);
			bottom  = int(bottom * height);
    	}
        float score = result[i + 2];
        int labelIndex = (int)(result[i + 1] + 0.000001);

        pubJob->addJobElem(Job::FloatType, 1, (void*)&top);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&left);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&bottom);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&right);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&score);
        pubJob->addJobElem(Job::IntType, 1, (void*)&labelIndex);
    }
    Broker::publish(job->getJobID(), pubJob);
}


void Worker::handleRunObjectDetectionNetworkWithInput(Job* job) {
    string networkID = job->getStringValue(0);
    int channel = job->getIntValue(1);
    int height = job->getIntValue(2);
    int width = job->getIntValue(3);
    int baseNetworkType = job->getIntValue(4);
    int useAdhocRun = job->getIntValue(5);
    float* imageData = job->getFloatArray(6);

    bool runNetwork = true;
    if (useAdhocRun)
        runNetwork = Network<float>::addAdhocRun(networkID);

    if (!runNetwork) {
        Job* pubJob = getPubJob(job);
        int error = -1;
        pubJob->addJobElem(Job::IntType, 1, (void*)&error);
        Broker::publish(job->getJobID(), pubJob);
        return;
    }

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    WorkContext::updateNetwork(networkID);
    
    InputLayer<float>* commonInputLayer;
    Layer<float>* commonOutputLayer;
    commonInputLayer = (InputLayer<float>*)network->findLayer(SNPROP(inputLayer), 
            LayerActivation::TestActivation);
    SASSUME0(commonInputLayer != NULL);
    commonOutputLayer = network->findLayer(SNPROP(outputLayer), LayerActivation::TestActivation);
    SASSUME0(commonOutputLayer != NULL);

    WorkContext::updateLayer(networkID, commonInputLayer->layerID);
    SASSUME0(baseNetworkType < (int)WORKER_OD_eMAX);

    if (!useAdhocRun) {
        commonInputLayer->feedImage(channel, height, width, imageData);
        network->runMiniBatch(true, 0);
    } else {
        network->runAdhoc(SNPROP(inputLayer), channel, height, width, imageData);
    }

    ThreadMgmt::wait(WorkContext::curThreadID, 0);

    if (useAdhocRun)
        Network<float>::removeAdhocRun(networkID);

    Job* pubJob = getPubJob(job);

    int count = commonOutputLayer->_outputData[0]->getCount();
    const float* result = commonOutputLayer->_outputData[0]->host_data();
    int resultCount = 0;

    for (int i = 0; i < count; i += 7) {
        resultCount++;
    }
    pubJob->addJobElem(Job::IntType, 1, (void*)&resultCount);

    float left, top, right, bottom;
    for (int i = 0; i < resultCount; i++) {
        if (baseNetworkType == WORKER_OD_eSSD || baseNetworkType == WORKER_OD_eYOLO) {
            // SSD , YOLO case : 여기서는 무조건 절대좌표로 변환
            left	= std::min(std::max(result[i * 7 + 3], 0.f), 1.f);
            top		= std::min(std::max(result[i * 7 + 4], 0.f), 1.f);
            right	= std::min(std::max(result[i * 7 + 5], 0.f), 1.f);
            bottom	= std::min(std::max(result[i * 7 + 6], 0.f), 1.f);

            left    = int(left * width);
            top     = int(top * height);
            right   = int(right * width);
            bottom  = int(bottom * height);
        } else {
            // frcnn case
		    SASSUME0(baseNetworkType == WORKER_OD_eFRCNN);
            left	= int(result[i * 7 + 3]);
            top		= int(result[i * 7 + 4]);
            right	= int(result[i * 7 + 5]);
            bottom	= int(result[i * 7 + 6]);
        }

        float score = result[i * 7 + 2];
        int labelIndex = (int)(result[i * 7 + 1] + 0.000001);

        pubJob->addJobElem(Job::FloatType, 1, (void*)&top);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&left);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&bottom);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&right);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&score);
        pubJob->addJobElem(Job::IntType, 1, (void*)&labelIndex);
    }

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleRunClassificationNetworkWithInput(Job* job) {
    string networkID = job->getStringValue(0);
    int channel = job->getIntValue(1);
    int height = job->getIntValue(2);
    int width = job->getIntValue(3);
    int baseNetworkType = job->getIntValue(4);
    int maxResultCount = job->getIntValue(5);
    int useAdhocRun = job->getIntValue(6);
    float* imageData = job->getFloatArray(7);

    bool runNetwork = true;
    if (useAdhocRun)
        runNetwork = Network<float>::addAdhocRun(networkID);

    if (!runNetwork) {
        Job* pubJob = getPubJob(job);
        int error = -1;
        pubJob->addJobElem(Job::IntType, 1, (void*)&error);
        Broker::publish(job->getJobID(), pubJob);
        return;
    }

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    WorkContext::updateNetwork(networkID);
   
    InputLayer<float>* commonInputLayer;
    Layer<float>* commonOutputLayer;
    commonInputLayer = (InputLayer<float>*)network->findLayer(SNPROP(inputLayer),
            LayerActivation::TestActivation);
    SASSUME0(commonInputLayer != NULL); 
    commonOutputLayer = network->findLayer(SNPROP(outputLayer), LayerActivation::TestActivation);
    SASSUME0(commonOutputLayer != NULL); 

    WorkContext::updateLayer(networkID, commonInputLayer->layerID);
    SASSUME0(baseNetworkType < (int)WORKER_IC_eMAX);

    if (!useAdhocRun) {
        commonInputLayer->feedImage(channel, height, width, imageData);
        network->runMiniBatch(true, 0);
    } else {
        network->runAdhoc(SNPROP(inputLayer), channel, height, width, imageData);
    }

    ThreadMgmt::wait(WorkContext::curThreadID, 0);

    if (useAdhocRun)
        Network<float>::removeAdhocRun(networkID);

    Job* pubJob = getPubJob(job);

    const float* result = commonOutputLayer->_outputData[0]->host_data();
    int count = commonOutputLayer->_outputData[0]->getCount();

    // FIXME: memory 상에서 exponential을 해주는 것이 속도상 부담이 될 경우에는
    //        GPU상에서 exponential을 해주도록 수정하자.
    float max = result[0]; 
    for (int i = 1; i < count; i++) {
        if (max < result[i])
            max = result[i];
    }

    vector<pair<float, int>> resultVec;
    float sum = 0.0;
    for (int i = 0; i < count; i++) {
        float expVal = exp(result[i] - max);
        resultVec.push_back(make_pair(expVal, i));
        sum += expVal;
    }

    for (int i = 0; i < count; i++) {
        resultVec[i].first = resultVec[i].first / sum;
    }

    sort(resultVec.begin(), resultVec.end());

    int resultCount = min(count, maxResultCount);
    pubJob->addJobElem(Job::IntType, 1, (void*)&resultCount);

    for (int i = 0; i < resultCount; i++) {
        pubJob->addJobElem(Job::IntType, 1, (void*)&resultVec[count - 1 - i].second);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&resultVec[count - 1 - i].first);
    }
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleStartIDP(Job* job) {
    string networkID = job->getStringValue(0);
    InputDataProvider::handleIDP(networkID);
}

void Worker::handleCheckNetworkDef(Job* job) {
    string networkDef = job->getStringValue(0);

    string networkID = "";
    int gpuMBSize = 0;

    PlanValidation ret = PlanValidator::check(networkDef, networkID, gpuMBSize);
    int layerID = -1;
    string message;
    
    if (ret != PLAN_VALIDATION_eSUCCESS && 
        ret != PLAN_VALIDATION_eJSON_PARSE_FAILED &&
        ret != PLAN_VALIDATION_eNOT_ENOUGH_GPU) {
        SASSUME0(networkID != "");

        NetworkEvent event;
        bool hasEvent = NetworkRecorder::getValidationEvent(networkID, event);

        if (hasEvent) {
            layerID = event.layerID;
            message = event.msg;
        } else {
            message = "unknown error";
        }
    } else if (ret == PLAN_VALIDATION_eJSON_PARSE_FAILED) {
        message = "JSON parse failed.";
    } else if (ret == PLAN_VALIDATION_eNOT_ENOUGH_GPU) {
        message = "Not enough GPU Memory.";
    } else {
        SASSUME0(ret == PLAN_VALIDATION_eSUCCESS);
        message = "success";
    }

    Job* pubJob = getPubJob(job);
    int errorCode;

    if (ret == PLAN_VALIDATION_eSUCCESS) {
        errorCode = 0;
    } else if (ret == PLAN_VALIDATION_eNOT_ENOUGH_GPU) {
        errorCode = 2;
    } else {
        errorCode = 1;
    }

    pubJob->addJobElem(Job::IntType, 1, (void*)&errorCode);
    pubJob->addJobElem(Job::IntType, 1, (void*)&gpuMBSize);
    pubJob->addJobElem(Job::IntType, 1, (void*)&layerID);
    pubJob->addJobElem(Job::StringType, strlen(message.c_str()), (void*)message.c_str());
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleGetNetworkProgress(Job* job) {
    string networkID = job->getStringValue(0);

    int totalIterCount;
    int curIterCount;

    WorkContext::getNetworkProgress(networkID, curIterCount, totalIterCount);
    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::IntType, 1, (void*)&curIterCount);
    pubJob->addJobElem(Job::IntType, 1, (void*)&totalIterCount);

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleGetNetworkResult(Job* job) {
    string networkID = job->getStringValue(0);
    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    WorkContext::updateNetwork(networkID);

    vector<string> itemNames;
    vector<float> itemResults;

    for (int i = 0; i < SNPROP(measureLayer).size(); i++) {
        string measureLayerName = SNPROP(measureLayer)[i];
        Layer<float>* layer = network->findLayer(measureLayerName);

        MeasureLayer<float>* measureLayer = 
            dynamic_cast<MeasureLayer<float>*>(layer);
        
        if (measureLayer == NULL)
            continue;

        float measureVal = measureLayer->measureAll();
        if (measureVal != measureVal) // NaN case
            measureVal = 0.0;

        itemNames.push_back(measureLayerName);
        itemResults.push_back(measureVal);
    }

    int measureCount = itemResults.size();
    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::IntType, 1, (void*)&measureCount);

    for (int i = 0; i < itemResults.size(); i++) {
        pubJob->addJobElem(Job::StringType, strlen(itemNames[i].c_str()),
                (void*)itemNames[i].c_str());
        pubJob->addJobElem(Job::FloatType, 1, (void*)&itemResults[i]);
    }
    Broker::publish(job->getJobID(), pubJob);
}

bool Worker::handleJob(Job* job) {
    bool doLoop = true;

    switch (job->getType()) {
        case JobType::HaltMachine:
            doLoop = false;
            ThreadMgmt::signalAll(ThreadEvent::Halt);
            break;

        case JobType::CreateNetworkFromFile:
            handleCreateNetworkFromFileJob(job);
            break;

        case JobType::CreateNetwork:
            handleCreateNetwork(job);
            break;

        case JobType::CreateResumeNetwork:
            handleCreateResumeNetwork(job);
            break;

        case JobType::StopNetwork:
            handleStopNetwork(job);
            break;

        case JobType::DestroyNetwork:
            handleDestroyNetwork(job);
            break;

        case JobType::BuildNetwork:
            handleBuildNetwork(job);
            break;

        case JobType::ResetNetwork:
            handleResetNetwork(job);
            break;

        case JobType::RunNetwork:
            handleRunNetwork(job);
            break;

        case JobType::RunNetworkMiniBatch:
            handleRunNetworkMiniBatch(job);
            break;

        case JobType::SaveNetwork:
            handleSaveNetwork(job);
            break;

        case JobType::LoadNetwork:
            handleLoadNetwork(job);
            break;

        case JobType::GetNetworkEvent:
            handleGetNetworkEvent(job);
            break;

        case JobType::GetNetworkEventMessage:
            handleGetNetworkEventMessage(job);
            break;

        case JobType::RunNetworkWithInputData:
            handleRunNetworkWithInputData(job);
            break;

        case JobType::RunObjectDetectionNetworkWithInput:
            handleRunObjectDetectionNetworkWithInput(job);
            break;

        case JobType::RunClassificationNetworkWithInput:
            handleRunClassificationNetworkWithInput(job);
            break;

        case JobType::StartInputDataProvider:
            handleStartIDP(job);
            break;

        case JobType::GetMeasureItemName:
            handleGetMeasureItemName(job);
            break;

        case JobType::GetMeasures:
            handleGetMeasures(job);
            break;

        case JobType::CheckNetworkDef:
            handleCheckNetworkDef(job);
            break;

        case JobType::GetNetworkProgress:
            handleGetNetworkProgress(job);
            break;

        case JobType::GetNetworkResult:
            handleGetNetworkResult(job);
            break;

        default:
            SASSERT(false, "Invalid job type");
    }

    return doLoop;
}

void Worker::launchThreads(int taskConsumerCount, int jobConsumerCount) {
    // (1) Cuda를 생성한다.
    Cuda::create(SPARAM(GPU_COUNT));
    COLD_LOG(ColdLog::INFO, true, "CUDA is initialized");

    // (2) Worker Count를 설정한다.
    if (taskConsumerCount > Cuda::gpuCount) {
        SYS_LOG("ERROR: Invalid GPU count of Worker. ");
        SYS_LOG("There are %d available GPU but requested GPU count of Worker is %d.",
            Cuda::gpuCount, taskConsumerCount);
        exit(1);
    }

	// (3) producer 쓰레드를 생성한다.
    Worker::producer = NULL;
    SNEW_ONCE(Worker::producer, thread, producerThread);
    SASSUME0(producerThread != NULL);

	// (4) consumer 쓰레드들을 생성한다.
	for (int i = 0; i < SPARAM(GPU_COUNT); i++) {
		Worker::consumers.push_back(thread(taskConsumerThread, i, Cuda::availableGPU[i]));
        TaskQueue *tq = NULL;
        SNEW_ONCE(tq, TaskQueue);
        SASSUME0(tq != NULL);
        Worker::taskQueues.push_back(tq);
    }

    for (int i = 0; i < SPARAM(JOB_CONSUMER_COUNT); i++) {
        Worker::consumers.push_back(thread(jobConsumerThread, i));
        Worker::jcReadyQueue.push_back(i);
    }
}

void Worker::joinThreads() {
	for (int i = 0; i < Worker::consumers.size(); i++) {
		Worker::consumers[i].join();
        // XXX: 그냥 쓰레드는 메모리해제 관련하여 아무 작업도 필요 없나? 확인해봐야 함!!
	}
    Worker::consumers.clear();

	Worker::producer->join();
	SDELETE(Worker::producer);
    Worker::producer = NULL;

    for (int i = 0; i < Worker::taskQueues.size(); i++) {
        SDELETE(Worker::taskQueues[i]);
    }

    Worker::taskQueues.clear();
}

int Worker::pushJob(Job* job) {
    int pubJobID = -1;

    // (1) pubJob이 있는 경우에 pubJob을 생성하고 pubJob ID를 할당받는다.
    if (job->hasPubJob()) {
        Job* pubJob = NULL;
        SNEW(pubJob, Job, job->getPubJobType());
        SASSUME0(pubJob != NULL);

        unique_lock<mutex> reqPubJobMapLock(Job::reqPubJobMapMutex); 
        Job::reqPubJobMap[job->getJobID()] = pubJob; 
        // subscriber will deallocate pubJob
        reqPubJobMapLock.unlock();
        pubJobID = pubJob->getJobID();
    }

    // (2) job queue에 job을 넣는다.
    Worker::jobQueueMutex.lock();
    Worker::jobQueue.push_back(job);
    Worker::jobQueueMutex.unlock();

    // (3) 프로듀서에게 새로운 잡이 추가되었음을 알려준다.
    ThreadMgmt::signal(ThreadMgmt::getThreadID(ThreadType::Producer, 0), ThreadEvent::Wakeup);

    return pubJobID;
}

Job* Worker::popJob() {
    Job* popedJob;
    Worker::jobQueueMutex.lock();
    
    if (Worker::jobQueue.empty()) {
        Worker::jobQueueMutex.unlock();
        return NULL;
    }

    popedJob = Worker::jobQueue.front();
    Worker::jobQueue.pop_front();
    Worker::jobQueueMutex.unlock();

    return popedJob;
}

int Worker::getJobCount() {
    unique_lock<mutex> lock(Worker::jobQueueMutex);
    return jobQueue.size();
}

void Worker::insertJCReadyQueue(int consumerIdx) {
    unique_lock<mutex> lock(Worker::jcReadyMutex);
    jcReadyQueue.push_back(consumerIdx);
}

vector<int> Worker::getReadyJCs(int count) {
    vector<int> result;
    unique_lock<mutex> lock(Worker::jcReadyMutex);
    for (int i = 0; i < count; i++) {
        if (Worker::jcReadyQueue.empty())
            break;

        int popedJCIdx = Worker::jcReadyQueue.front();
        Worker::jcReadyQueue.pop_front();
        result.push_back(popedJCIdx);
    }
    lock.unlock();
    return result;
}

TaskAllocTensor* Worker::addAllocTensorTask(int consumerIdx, int nodeID, int devID,
    int requestThreadID, string tensorName) {
    TaskAllocTensor* task = (TaskAllocTensor*)Task::getElem(TaskType::AllocTensor);
    SASSUME0(task != NULL);     // pool이 넉넉하지 않을때에 대한 전략이 반드시 필요하다

    task->nodeID = nodeID;
    task->devID = devID;
    task->requestThreadID = requestThreadID;
    task->tensorName = tensorName;
    task->step = TaskAllocTensorStep::Alloc;

    SASSUME0(consumerIdx < Worker::taskQueues.size());
    TaskQueue* tq = Worker::taskQueues[consumerIdx];

    unique_lock<mutex> lock(tq->mutex);
    tq->tasks.push_back((TaskBase*)task);

    return task;
}

void Worker::addRunPlanTask(int consumerIdx, string networkID, int dopID, bool inference,
    int requestThreadID, bool useAdhocRun, string inputLayerName, int channel, int height, 
    int width, float* imageData) {

    TaskRunPlan* task = (TaskRunPlan*)Task::getElem(TaskType::RunPlan);
    SASSUME0(task != NULL);     // pool이 넉넉하지 않을때에 대한 전략이 반드시 필요하다

    task->networkID = networkID;
    task->dopID = dopID;
    task->inference = inference;
    task->requestThreadID = requestThreadID;
    task->inputLayerName = inputLayerName;
    task->useAdhocRun = useAdhocRun;
    task->channel = channel;
    task->height = height;
    task->width = width;
    task->imageData = imageData;

    SASSUME0(consumerIdx < Worker::taskQueues.size());
    TaskQueue* tq = Worker::taskQueues[consumerIdx];

    unique_lock<mutex> lock(tq->mutex);
    tq->tasks.push_back((TaskBase*)task);
}

void Worker::addUpdateTensorTask(int consumerIdx, string networkID, int dopID, int layerID,
    int planID, vector<UpdateParam> updateParams) {

    TaskUpdateTensor* task = (TaskUpdateTensor*)Task::getElem(TaskType::UpdateTensor);
    SASSUME0(task != NULL);

    task->networkID = networkID;
    task->dopID = dopID;
    task->layerID = layerID;
    task->planID = planID;

    for (int i = 0 ; i < updateParams.size(); i++)
        task->updateParams.push_back(updateParams[i]);

    SASSUME0(consumerIdx < Worker::taskQueues.size());
    TaskQueue* tq = Worker::taskQueues[consumerIdx];

    unique_lock<mutex> lock(tq->mutex);
    tq->tasks.push_back((TaskBase*)task);
}

void Worker::addAllocLayerTask(int consumerIdx, string networkID, int dopID, int layerID,
    int nodeID, int devID, int requestThreadID, int layerType, void* instancePtr) {
    
    TaskAllocLayer* task = (TaskAllocLayer*)Task::getElem(TaskType::AllocLayer);
    SASSUME0(task != NULL);

    task->networkID = networkID;
    task->dopID = dopID;
    task->layerID = layerID;
    task->nodeID = nodeID;
    task->devID = devID;
    task->requestThreadID = requestThreadID;
    task->layerType = layerType;
    task->instancePtr = instancePtr;

    SASSUME0(consumerIdx < Worker::taskQueues.size());
    TaskQueue* tq = Worker::taskQueues[consumerIdx];

    unique_lock<mutex> lock(tq->mutex);
    tq->tasks.push_back((TaskBase*)task);
}

int Worker::getConsumerIdx(int devIdx) {
    for (int i = 0; i < Cuda::availableGPU.size(); i++) {
        if (Cuda::availableGPU[i] == devIdx)
            return i;
    }

    return -1;
}
