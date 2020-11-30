/**
 * @file RunNetworkTest.cpp
 * @date 2017-06-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string>
#include <iostream>

#include "RunNetworkTest.h"
#include "common.h"
#include "StdOutLog.h"
#include "ClientAPI.h"
#include "Communicator.h"
#include "SysLog.h"

using namespace std;

#if 1
#define NETWORK_FILEPATH       SPATH("plan/test/network.conf.test")
#else
//#define NETWORK_FILEPATH       SPATH("examples/LeNet/lenet_train.json")
#endif

bool RunNetworkTest::runSimpleTest() {
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

    ret = ClientAPI::runNetwork(handle, netHandle, false);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::destroyNetwork(handle, netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::releaseSession(handle);
    SASSERT0(ret == ClientError::Success);

    return true;
}


bool RunNetworkTest::runMiniBatchTest() {
    ClientError ret;
    ClientHandle handle;
    NetworkHandle netHandle;

    ret = ClientAPI::createHandle(handle, "localhost", Communicator::LISTENER_PORT); 
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::getSession(handle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::createNetworkFromFile(handle, string(NETWORK_FILEPATH), netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::buildNetwork(handle, netHandle, 1);
    SASSERT0(ret == ClientError::Success);

    for (int i = 0 ; i < 3; i++) {
        ret = ClientAPI::runNetworkMiniBatch(handle, netHandle, false, i);
        if (ret != ClientError::Success) {
            cout << "run minibatch failed." << endl;
        }
    }

    ret = ClientAPI::destroyNetwork(handle, netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::releaseSession(handle);
    SASSERT0(ret == ClientError::Success);

    return true;
}

bool RunNetworkTest::runTwiceTest() {
    ClientError ret;
    ClientHandle handle;
    NetworkHandle netHandle;

    ret = ClientAPI::createHandle(handle, "localhost", Communicator::LISTENER_PORT); 
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::getSession(handle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::createNetworkFromFile(handle, string(NETWORK_FILEPATH), netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::buildNetwork(handle, netHandle, 2);
    SASSERT0(ret == ClientError::Success);

    ret = ClientAPI::runNetwork(handle, netHandle, false);
    SASSERT0(ret == ClientError::Success);

    ret = ClientAPI::resetNetwork(handle, netHandle);
    SASSERT0(ret == ClientError::Success);

    ret = ClientAPI::runNetwork(handle, netHandle, false);
    SASSERT0(ret == ClientError::Success);

    ret = ClientAPI::destroyNetwork(handle, netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::releaseSession(handle);
    SASSERT0(ret == ClientError::Success);

    return true;
}

bool RunNetworkTest::runTest() {
    bool result = runSimpleTest();
    if (result) {
        STDOUT_LOG("*  - simple run network test is success");
    } else {
        STDOUT_LOG("*  - simple run network test is failed");
        return false;
    }

    result = runMiniBatchTest();
    if (result) {
        STDOUT_LOG("*  - run network minibatch test is success");
    } else {
        STDOUT_LOG("*  - run network minibatch test is failed");
        return false;
    }

    result = runTwiceTest();
    if (result) {
        STDOUT_LOG("*  - run network twice test is success");
    } else {
        STDOUT_LOG("*  - run network twice test is failed");
        return false;
    }

    return true;
}
