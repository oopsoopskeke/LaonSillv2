/**
 * @file ThreadMgmt.cpp
 * @date 2017-06-08
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <chrono>

#include "ThreadMgmt.h"
#include "SysLog.h"
#include "WorkContext.h"
#include "MemoryMgmt.h"

using namespace std;

int ThreadMgmt::threadCount;
vector<ThreadContext*> ThreadMgmt::contextArray;
int* ThreadMgmt::threadIDBaseArray;
volatile atomic<int> ThreadMgmt::readyCount;

int ThreadMgmt::init() {
    ThreadMgmt::threadCount = 1 +   /* producer count */
        SPARAM(GPU_COUNT) +  /* task consumer count */
        SPARAM(JOB_CONSUMER_COUNT) +    /* job consumer count */
        2;  /* network sender & receiver */

    for (int i = 0; i < ThreadMgmt::threadCount; i++) {
        ThreadContext* tc = NULL;
        SNEW_ONCE(tc, ThreadContext);
        SASSERT0(tc != NULL);
        tc->threadID = i;
        atomic_store(&tc->event, 0UL);
        ThreadMgmt::contextArray.push_back(tc);
    }

    int allocSize = sizeof(int) * ThreadType::ThreadTypeMax;
    SMALLOC_ONCE(ThreadMgmt::threadIDBaseArray, int, allocSize);
    SASSERT0(ThreadMgmt::threadIDBaseArray != NULL);

    // threadIDBaseArray와 이름을 채운다.
    int offset = 0;
    ThreadMgmt::threadIDBaseArray[ThreadType::Producer] = offset;
    ThreadMgmt::contextArray[offset]->name = "Producer";
    offset++;

    ThreadMgmt::threadIDBaseArray[ThreadType::TaskConsumer] = offset;
    for (int i = 0; i < SPARAM(GPU_COUNT); i++) {
        ThreadMgmt::contextArray[offset]->name = string("TaskConsumer#") + to_string(i);
        offset++;
    }

    ThreadMgmt::threadIDBaseArray[ThreadType::JobConsumer] = offset;
    for (int i = 0; i < SPARAM(JOB_CONSUMER_COUNT); i++) {
        ThreadMgmt::contextArray[offset]->name = string("JobConsumer#") + to_string(i);
        offset++;
    }

    ThreadMgmt::threadIDBaseArray[ThreadType::Sender] = offset;
    ThreadMgmt::contextArray[offset]->name = "Sender";
    offset++;

    ThreadMgmt::threadIDBaseArray[ThreadType::Receiver] = offset;
    ThreadMgmt::contextArray[offset]->name = "Receiver";
    offset++;

    SASSERT0(offset == ThreadMgmt::threadCount);
    atomic_store(&ThreadMgmt::readyCount, 0);

    return ThreadMgmt::threadCount;
}

void ThreadMgmt::destroy() {
    vector<ThreadContext*>::iterator iter = ThreadMgmt::contextArray.begin();
    for(; iter != ThreadMgmt::contextArray.end(); ++iter) {
        SDELETE(*iter);
    }

    ThreadMgmt::contextArray.clear();

    SASSERT0(ThreadMgmt::threadIDBaseArray != NULL);
    SFREE(ThreadMgmt::threadIDBaseArray);
}

void ThreadMgmt::signal(int threadID, ThreadEvent event) {
    SASSUME0(threadID < ThreadMgmt::threadCount);
    ThreadContext* context = ThreadMgmt::contextArray[threadID];
    unique_lock<mutex> lock(context->mutex);
    atomic_fetch_or(&context->event, (unsigned long)event);
    context->flag = true;
    context->cv.notify_one();
}

void ThreadMgmt::signalAll(ThreadEvent event) {
    for (int i = 0; i < ThreadMgmt::threadCount; i++) {
        ThreadContext* context = ThreadMgmt::contextArray[i];
        unique_lock<mutex> lock(context->mutex);
        atomic_fetch_or(&context->event, (unsigned long)event);
        context->flag = true;
        context->cv.notify_one();
        lock.unlock();
    }
}

ThreadEvent ThreadMgmt::wait(int threadID, unsigned long timeout) {
    SASSUME0(threadID < ThreadMgmt::threadCount);
    ThreadContext* context = ThreadMgmt::contextArray[threadID];
    if (context->flag) {
        context->flag = false;
        ThreadEvent event = (ThreadEvent)atomic_exchange(&context->event, (unsigned long)0UL);

        if (event == 0UL)
            return ThreadEvent::Wakeup;
        else
            return event;
    }
    unique_lock<mutex> lock(context->mutex);
    bool getEventInTime = true;
    if (timeout == 0UL) {
        context->cv.wait(lock);
    } else {
        cv_status status = context->cv.wait_for(lock, chrono::milliseconds(timeout));
        if (status == cv_status::timeout) {
            getEventInTime = false;
        }
    }
    context->flag = false;

    if (!getEventInTime) {
        atomic_fetch_or(&context->event, (unsigned long)ThreadEvent::Timeout);
    }

    ThreadEvent event = (ThreadEvent)atomic_exchange(&context->event, (unsigned long)0UL);

    return event;
}

int ThreadMgmt::getThreadID(ThreadType type, int index) {
    SASSUME0(type < ThreadType::ThreadTypeMax);
    int threadID = ThreadMgmt::threadIDBaseArray[type] + index;
    SASSUME0(threadID < ThreadMgmt::threadCount);
    return threadID;
}

bool ThreadMgmt::isReady() {
    int readyCount = atomic_load(&ThreadMgmt::readyCount);

    if (readyCount != ThreadMgmt::threadCount)
        return false;
    
    return true;
}

void ThreadMgmt::setThreadReady(int threadID) {
    WorkContext::curThreadID = threadID;
    atomic_fetch_add(&ThreadMgmt::readyCount, 1); 
}
