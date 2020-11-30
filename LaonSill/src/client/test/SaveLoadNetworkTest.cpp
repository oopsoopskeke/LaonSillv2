/**
 * @file SaveLoadNetworkTest.cpp
 * @date 2017-06-23
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "SaveLoadNetworkTest.h"
#include "common.h"
#include "StdOutLog.h"
#include "ClientAPI.h"
#include "Communicator.h"
#include "SysLog.h"

using namespace std;

#define NETWORK_FILEPATH       SPATH("plan/test/network.conf.test")

bool SaveLoadNetworkTest::runSaveTest() {
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

    ret = ClientAPI::saveNetwork(handle, netHandle, "test.param");
    SASSERT0(ret == ClientError::Success);

    ret = ClientAPI::destroyNetwork(handle, netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::releaseSession(handle);
    SASSERT0(ret == ClientError::Success);

    return true;
}

bool SaveLoadNetworkTest::runLoadTest() {
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

    ret = ClientAPI::loadNetwork(handle, netHandle, "test.param");
    SASSERT0(ret == ClientError::Success);

    ret = ClientAPI::runNetwork(handle, netHandle, false);
    SASSERT0(ret == ClientError::Success);

    ret = ClientAPI::destroyNetwork(handle, netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::releaseSession(handle);
    SASSERT0(ret == ClientError::Success);

    return true;
}

bool SaveLoadNetworkTest::runTest() {
    bool result = runSaveTest();
    if (result) {
        STDOUT_LOG("*  - save network test is success");
    } else {
        STDOUT_LOG("*  - save network test is failed");
        return false;
    }

    result = runLoadTest();
    if (result) {
        STDOUT_LOG("*  - load network test is success");
    } else {
        STDOUT_LOG("*  - load network test is failed");
        return false;
    }
    return true;
}
