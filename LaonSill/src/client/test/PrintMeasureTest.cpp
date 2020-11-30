/**
 * @file PrintMeasureTest.cpp
 * @date 2017-11-06
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string>
#include <iostream>
#include <thread>

#include "PrintMeasureTest.h"
#include "common.h"
#include "StdOutLog.h"
#include "ClientAPI.h"
#include "Communicator.h"
#include "SysLog.h"

using namespace std;

#define NETWORK_FILEPATH       SPATH("examples/LeNet/lenet_train.json")

static void printMeasureThread(string networkID) {
    ClientError ret;
    ClientHandle handle;

    cout << "now print measure thread starts" << endl;

    // (1) 새로운 세션으로 접속한다.
    ret = ClientAPI::createHandle(handle, "localhost", Communicator::LISTENER_PORT); 
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::getSession(handle);
    SASSERT0(ret == ClientError::Success);

    // (2) measure 아이템 리스트를 얻는다.
    vector<string> measureItemNames;
    ret = ClientAPI::getMeasureItemName(handle, networkID, measureItemNames);
    cout << "Measure Item Names : " << endl;
    for (int i = 0; i < measureItemNames.size(); i++) {
        cout << " - " << measureItemNames[i] << endl;
    }

    // (3) measure 데이터를 10건씩 5회 얻는다.
    int measureItemCount = measureItemNames.size();
    float* floatData = (float*)malloc(sizeof(float) * 10 * measureItemCount);
    SASSERT0(floatData != NULL);

    cout << " (1) forward search" << endl;
    for (int i = 0; i < 5; i++) {
        int startIterNum;
        int dataCount;
        int curIterNum;
        int totalIterNum;

        ret = ClientAPI::getMeasures(handle, networkID, true, i * 10, 10, &startIterNum,
                &dataCount, &curIterNum, &totalIterNum, floatData); 

        for (int j = 0; j < dataCount / measureItemCount; j++) {
            cout << "Measure Data[" << startIterNum + j << "] : (";
            bool isFirst = true;
            for (int k = 0; k < measureItemCount; k++) {
                if (isFirst) {
                    isFirst = false;
                } else {
                    cout << ",";
                }
                cout << floatData[j * measureItemCount + k];
            }
            cout << ")" << endl;
        }
    }

    // (4) measure 데이터를 40건씩 최신 데이터를 얻는다.
    cout << " (2) reverse search" << endl;
    for (int i = 0; i < 5; i++) {
        int startIterNum;
        int dataCount;
        int curIterNum;
        int totalIterNum;

        ret = ClientAPI::getMeasures(handle, networkID, false, i * 10, 10, &startIterNum,
                &dataCount, &curIterNum, &totalIterNum, floatData); 

        for (int j = 0; j < dataCount / measureItemCount; j++) {
            cout << "Measure Data[" << startIterNum + j << "] : (";
            bool isFirst = true;
            for (int k = 0; k < measureItemCount; k++) {
                if (isFirst) {
                    isFirst = false;
                } else {
                    cout << ",";
                }
                cout << floatData[j * measureItemCount + k];
            }
            cout << ")" << endl;
        }
    }

    free(floatData);

    ret = ClientAPI::releaseSession(handle);
    SASSERT0(ret == ClientError::Success);

    cout << "measure thread ends" << endl;
}

bool PrintMeasureTest::runSimpleTest() {
    ClientError ret;
    ClientHandle handle;
    NetworkHandle netHandle;

    ret = ClientAPI::createHandle(handle, "localhost", Communicator::LISTENER_PORT); 
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::getSession(handle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::createNetworkFromFile(handle, string(NETWORK_FILEPATH), netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::buildNetwork(handle, netHandle, 4);
    SASSERT0(ret == ClientError::Success);

    thread *measureThread = new thread(printMeasureThread, netHandle.networkID);

    ret = ClientAPI::runNetwork(handle, netHandle, false);
    SASSERT0(ret == ClientError::Success);

    measureThread->join();
    delete measureThread;

    ret = ClientAPI::destroyNetwork(handle, netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::releaseSession(handle);
    SASSERT0(ret == ClientError::Success);

    return true;
}

bool PrintMeasureTest::runTest() {
    bool result = runSimpleTest();
    if (result) {
        STDOUT_LOG("*  - simple measure test is success");
    } else {
        STDOUT_LOG("*  - simple measure test is failed");
        return false;
    }

    return true;
}
