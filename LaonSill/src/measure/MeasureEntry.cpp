/**
 * @file MeasureEntry.cpp
 * @date 2017-11-02
 * @author moonhoen lee
 * @brief 
 *  정확도, 에러율과 같은 관측가능한 수치들을 관리하는 모듈이다.
 * @details
 *  옵션으로 file 관리와 memory 관리를 지원하는데, 현재는 memory 관리만 지원하고 있다.
 *  추후에 수정할 예정이다.
 */

#include "MeasureEntry.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "MemoryMgmt.h"
#include "PropMgmt.h"

using namespace std;

#define MEASURE_ENTRY_ADDDATA_MAX_RETRY_COUNT   10
#define MEASURE_ENTRY_RETRY_MSEC                10UL

extern const char* LAONSILL_HOME_ENVNAME;

MeasureEntry::MeasureEntry(string networkID, int queueSize, MeasureOption option,
    vector<string> itemNames) {
    this->networkID = networkID;
    this->queueSize = queueSize;
    this->itemNames = itemNames;
    this->itemCount = this->itemNames.size();
    SASSERT0(this->itemCount > 0);

    // 큐 사이즈가 아이템 개수의 배수가 되도록 설정한다.
    // (계산을 간단하게 하기 위해서)
    int remain = this->queueSize % this->itemCount;
    this->queueSize = this->queueSize - remain + this->itemCount;

    this->head = 0;
    this->option = option;

    this->baseIterNum = 0;
    this->refCount = 0;

    this->data = NULL;
    int allocSize = sizeof(float) * this->queueSize * this->itemCount;
    SMALLOC(this->data, float, allocSize);
    SASSERT0(this->data != NULL);

    this->status = NULL;
    allocSize = sizeof(MeasureEntryDataStatus) * this->queueSize;
    SMALLOC(this->status, MeasureEntryDataStatus, allocSize);
    SASSERT0(this->status != NULL);
    for (int i = 0; i < this->queueSize; i++)
        this->status[i] = MEASURE_ENTRY_STATUS_NONE;

    this->readRefCount = NULL;
    allocSize = sizeof(int) * this->queueSize;
    SMALLOC(this->readRefCount, int, allocSize);
    SASSERT0(this->readRefCount != NULL);
    for (int i = 0; i < this->queueSize; i++)
        this->readRefCount[i] = 0;

    allocSize = sizeof(float) * this->itemCount;
    SMALLOC(this->addBuffer, float, allocSize);
    SASSERT0(this->addBuffer != NULL);

    if (option & MEASURE_OPTION_FILE) {
        char measureFilePath[PATH_MAX];

        sprintf(measureFilePath, "%s/measure/%s.measure", getenv(LAONSILL_HOME_ENVNAME),
            networkID.c_str());
        this->fp = fopen(measureFilePath, "w+");
        SASSERT0(this->fp != NULL);
    } else {
        this->fp = NULL;
    }
}

MeasureEntry::~MeasureEntry() {
    SASSERT0(this->data != NULL);
    SFREE((void*)this->data);

    SASSERT0(this->status != NULL);
    SFREE((void*)this->status);

    SASSERT0(this->readRefCount != NULL);
    SFREE((void*)this->readRefCount);

    SASSERT0(this->addBuffer != NULL);
    SFREE(this->addBuffer);

    if (this->fp != NULL) {
        fflush(this->fp);
        fclose(this->fp);
    }
}

// XXX: 코딩을 하고 나니 자원보호가 조금 비효율적으로 되어 있어 보인다. 
//      겨우 메모리카피 하는 부분에 대해서 보호하려고 RWLock과 ref count를 두는 것이
//      맞는 방식은 아닌 것 같다. 추후에 수정하자.
void MeasureEntry::addData(float* data) {
    int dataOffset;

    // write access는 언제나 한명이 하나의 원소에 대해서만 수행하게 될 것이다.
    // 아니면 아키텍쳐를 수정해야 한다.
    dataOffset = this->head * this->itemCount;

    // FIXME: Mutex로 자원보호를 하는 것보다는 AOP로 하는 것이 성능상 더 좋아보니다.
    bool doLoop = true;
    int retryCount = 0;
    unique_lock<mutex> entryLock(this->entryMutex);
    while (doLoop) {
        MeasureEntryDataStatus curStatus = this->status[this->head];

        if (curStatus == MEASURE_ENTRY_STATUS_WRITE) {
            entryLock.unlock();
            SASSERT0(false);
        } else if (curStatus == MEASURE_ENTRY_STATUS_READ) {
            entryLock.unlock();
            retryCount++;

            SASSERT(retryCount < MEASURE_ENTRY_ADDDATA_MAX_RETRY_COUNT,
                "can not add data in the measure entry(network ID=%s)",
                this->networkID.c_str());

            usleep(MEASURE_ENTRY_RETRY_MSEC);
            entryLock.lock();
        } else {        // MEASURE_ENTRY_STATUS_NONE case
            this->status[this->head] = MEASURE_ENTRY_STATUS_WRITE;
            entryLock.unlock();
            doLoop = false;
        }
    }

    memcpy((void*)&this->data[dataOffset], data, sizeof(float) * this->itemCount);

    // XXX: 지금은 file write를 sync하게 처리하였다. 큰 부담이 없을 것이라 생각해서
    // 진행하였으나 추후에 이곳에 부하가 많이 걸린다면 수정이 필요하다.
    if (this->fp != NULL) {
        fprintf(this->fp, "%d", (int)(this->head + this->baseIterNum + SNPROP(startIterNum)));
        for (int i = 0; i < this->itemCount; i++) {
            fprintf(this->fp, ",%f", data[i]);
        }
        fprintf(this->fp, "\n");
        fflush(this->fp);
    }

    entryLock.lock();
    this->status[this->head] = MEASURE_ENTRY_STATUS_NONE;
    this->head = this->head + 1;
    if (this->head == this->queueSize) {
        this->head = 0;
        this->baseIterNum = this->baseIterNum + this->queueSize;
    }
    entryLock.unlock();
}

void MeasureEntry::logHistoryData(vector<pair<int, vector<float>>> measures) {
    if (this->fp == NULL)
        return;

    for (int i = 0; i < measures.size(); i++) {
        fprintf(this->fp, "%d", measures[i].first);
        for (int j = 0; j < measures[i].second.size(); j++) {
            fprintf(this->fp, ",%f",measures[i].second[j]);
        }
        fprintf(this->fp, "\n");
    }
    fflush(this->fp);
}

void MeasureEntry::setAreaLock(int start, int count) {
    SASSUME0(count > 0);
    SASSUME0(start < this->queueSize);
    SASSUME0(start + count <= this->queueSize);

    for (int i = 0; i < count; i++) {
        int index = start + i;
        int retryCount = 0;

    // 복잡한 조건의 loop에서 goto를 통한 예외처리는 좋은 방법이 될 수 있다.
    // (혹시 이 소스를 보고 비판할까봐.....)
retry:
        MeasureEntryDataStatus curStatus = this->status[index];
        
        if (curStatus == MEASURE_ENTRY_STATUS_READ) {
            SASSUME0(this->readRefCount[index] > 0);
            this->readRefCount[index] = this->readRefCount[index] + 1;
        } 
        
        else if (curStatus == MEASURE_ENTRY_STATUS_WRITE) {
            this->entryMutex.unlock();
            retryCount++;

            SASSERT(retryCount < MEASURE_ENTRY_ADDDATA_MAX_RETRY_COUNT,
                "can not get data in the measure entry(network ID=%s)", 
                this->networkID.c_str());

            usleep(MEASURE_ENTRY_RETRY_MSEC);
            this->entryMutex.lock();

            goto retry;

        } else {        // MEASURE_ENTRY_STATUS_NONE case
            this->status[index] = MEASURE_ENTRY_STATUS_READ;
            SASSUME0(this->readRefCount[index] == 0);
            this->readRefCount[index] = 1;
        }
    }
}

void MeasureEntry::getDataInternal(int start, int count, float* data) {
    int dataOffset;
    dataOffset = start * this->itemCount;

    SASSUME0(count > 0);
    SASSUME0(start < this->queueSize);
    SASSUME0(start + count <= this->queueSize);
    SASSUME0(data != NULL);

    // XXX: 현재 메모리 기반의 Measure 모듈은 실제 필요한 동작이 메모리 카피 하나밖에 
    // 없는데 비해서 매우 많은 동기화 코드가 존재한다. 매우 잘못된 구현이다.
    // 하지만, 나중에 디스크로 flush하는 부분까지 생각을 한다면 괜찮을 수 있다.
    // 추후에 확인하여 더 나은 코드로 대체하자.
    memcpy((void*)data, (void*)&this->data[dataOffset],
        sizeof(float) * this->itemCount * count);

    unique_lock<mutex> entryLock(this->entryMutex);

    for (int i = 0; i < count; i++) {
        int index = start + i;
        int retryCount = 0;

        this->readRefCount[index] = this->readRefCount[index] - 1;
        if (this->readRefCount[index] == 0) {
            this->status[index] = MEASURE_ENTRY_STATUS_NONE;
        }
    }
    entryLock.unlock();
}

void MeasureEntry::getData(int start, int count, bool forward, int* startIterNum,
    int* measureCount, float* data) {
    int start1, start2;
    int count1, count2;

    if (count == 0) {
        (*measureCount) = 0;
        (*startIterNum) = -1;
        return;
    }

    SASSUME0(count < this->queueSize);

    // FIXME: Mutex로 자원보호를 하는 것보다는 AOP로 하는 것이 성능상 더 좋아보니다.
    unique_lock<mutex> entryLock(this->entryMutex);

    if (forward) {
        int queueIterNumBegin;      // Queue에서 가지고 있는 데이터의 최초 값
        int queueIterNumEnd;        // Queue에서 가지고 있는 데이터의 종료 값 + 1

        if (this->baseIterNum == 0) {
            queueIterNumBegin = 0;
        } else {
            queueIterNumBegin = (this->head + 1) % this->queueSize + 
                this->baseIterNum - this->queueSize;
        }
        queueIterNumEnd = this->head + this->baseIterNum;

        if ((queueIterNumBegin > queueIterNumEnd) ||
            (start > queueIterNumEnd) ||
            (start + count < queueIterNumBegin)) {
            entryLock.unlock();
            (*measureCount) = 0;
            (*startIterNum) = -1;   // dummy

            return;
        }

        int modifiedStart = max(start, queueIterNumBegin);
        int modifiedCount = min(start + count, queueIterNumEnd) - modifiedStart;

        start1 = modifiedStart % this->queueSize;

        if (this->queueSize - start1 < modifiedCount) {
            count1 = this->queueSize - start1;
            count2 = modifiedCount - count1;
            start2 = 0;
        } else {
            count1 = modifiedCount;
            count2 = 0;
            start2 = -1;    // meaning less
        }
        (*startIterNum) = modifiedStart;
    } else {
        if (this->head < count) {
            if (this->baseIterNum > 0) {
                start2 = 0;
                count2 = this->head;

                count1 = count - count2;
                start1 = this->queueSize - count1;
            } else {
                start1 = 0;
                count1 = this->head;
                start2 = -1;
                count2 = 0;
            }
        } else {
            start1 = this->head - count;
            count1 = count;
            start2 = -1;    //meaningless
            count2 = 0;
        }
        (*startIterNum) = this->baseIterNum + start1;
    }

    (*measureCount) = count1 + count2;

    if (count1 > 0)
        setAreaLock(start1, count1);

    if (count2 > 0)
        setAreaLock(start2, count2);

    entryLock.unlock();

    if (count1 > 0) {
        getDataInternal(start1, count1, data);
    }

    if (count2 > 0) {
        int offset = max(0, count1 * this->itemCount);
        getDataInternal(start2, count2, (float*)&data[offset]);
    }
}

void MeasureEntry::printStatus() {
    STDOUT_LOG("[Measure Entry Info]");
    STDOUT_LOG("  - networkID : %s", this->networkID.c_str());
    STDOUT_LOG("  - option : %d", int(this->option));
    STDOUT_LOG("  - queue size : %d", this->queueSize);

    STDOUT_LOG("  - item count : %d", this->itemCount);
    STDOUT_LOG("  - item names :");
    for (int i = 0 ; i < this->itemNames.size(); i++) {
        STDOUT_LOG("      > %s", this->itemNames[i].c_str());
    }
    STDOUT_LOG("  - head : %d", this->head);
    STDOUT_LOG("  - base iter num : %d", this->baseIterNum);
}
