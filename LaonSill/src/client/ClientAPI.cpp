/**
 * @file ClientAPI.cpp
 * @date 2017-06-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <stdlib.h>
#include <errno.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>

#include "ClientAPI.h"
#include "Communicator.h"
#include "MsgSerializer.h"
#include "MessageHeader.h"
#include "SysLog.h"
#include "Client.h"
#include "MemoryMgmt.h"

using namespace std;

#define CPPAPI_CHECK_BUFFER(job)                                                        \
    do {                                                                                \
        if (handle.buffer == NULL) {                                                    \
            close(handle.sockFD);                                                       \
            return ClientError::ClientHandleBufferAllocationFailed;                     \
        }                                                                               \
                                                                                        \
        Job* myJob = (Job*) job ;                                                       \
        int bufSize = MessageHeader::MESSAGE_HEADER_SIZE + myJob->getJobSize();         \
        if (bufSize > MessageHeader::MESSAGE_DEFAULT_SIZE) {                            \
            bigBuffer = (char*)malloc(bufSize);                                         \
            if (bigBuffer == NULL) {                                                    \
                close(handle.sockFD);                                                   \
                return ClientError::ClientHandleBufferAllocationFailed;                 \
            }                                                                           \
        }                                                                               \
    } while (0)

#define CPPAPI_CHECK_SEND()                                                             \
    do {                                                                                \
        if (ret != ClientError::Success) {                                              \
            if (bigBuffer != NULL)                                                      \
                free(bigBuffer);                                                        \
            return ClientError::SendJobFailed;                                          \
        }                                                                               \
    } while (0)

#define CPPAPI_CHECK_RECV(job, bufSize)                                                 \
    do {                                                                                \
        if (ret == Communicator::RecvOnlyHeader) {                                      \
            if (bigBuffer)                                                              \
                free(bigBuffer);                                                        \
                                                                                        \
            bigBuffer = (char*)malloc(bufSize);                                         \
                                                                                        \
            if (bigBuffer == NULL) {                                                    \
                close(handle.sockFD);                                                   \
                return ClientError::ClientHandleBufferAllocationFailed;                 \
            }                                                                           \
            ret = Client::recvJob(handle.sockFD, bigBuffer, & job, & bufSize);          \
        } else if (ret != Communicator::Success) {                                      \
            if (bigBuffer != NULL)                                                      \
                free(bigBuffer);                                                        \
            close(handle.sockFD);                                                       \
            return ClientError::RecvJobFailed;                                          \
        }                                                                               \
    } while (0)

// FIXME: 중복된 코드는 모아서 함수로 정의해서 사용하자.

ClientError ClientAPI::createHandle(ClientHandle& handle, std::string serverHostName,
    int serverPortNum) {

    if (strlen(serverHostName.c_str()) >= MAX_SERVER_HOSTNAME_LEN) {
        return ClientError::TooLongServerHostName;
    }

    handle.sockFD = 0;
    strcpy(handle.serverHostName, serverHostName.c_str());
    handle.serverPortNum = serverPortNum;
    handle.hasSession = false;

    return ClientError::Success;
}

ClientError ClientAPI::getSession(ClientHandle &handle) {
    if (handle.hasSession)
        return ClientError::HaveSessionAlready;

    // (1) get server info (struct hostent)
    struct hostent *server;
    server = gethostbyname(handle.serverHostName);
    if (server == NULL) {
        return ClientError::NoSuchHost;
    }

    // (2) create socket & connect to the server
    handle.sockFD = socket(AF_INET, SOCK_STREAM, 0);
    SASSERT0(handle.sockFD != -1);
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(struct sockaddr_in));
    serverAddr.sin_family = AF_INET;
    memcpy((char*)&serverAddr.sin_addr.s_addr, (char*)server->h_addr, server->h_length);
    serverAddr.sin_port = htons(handle.serverPortNum);

    int connectRetry = Client::connectRetry(handle.sockFD, (struct sockaddr*)&serverAddr,
                                            sizeof(serverAddr));
    if (connectRetry == -1) {
        return ClientError::HostConnectionFailed;
    }

    // (3-1) send welcome msg
    handle.buffer = (char*)malloc(MessageHeader::MESSAGE_DEFAULT_SIZE);
    if (handle.buffer == NULL) {
        close(handle.sockFD);
        return ClientError::ClientHandleBufferAllocationFailed;
    }
    handle.bufLen = MessageHeader::MESSAGE_DEFAULT_SIZE;

    MessageHeader msgHdr;
    msgHdr.setMsgType(MessageHeader::Welcome);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    MsgSerializer::serializeMsgHdr(msgHdr, handle.buffer);
    Communicator::CommRetType ret = Communicator::sendMessage(handle.sockFD, msgHdr,
        handle.buffer);
    if (ret != Communicator::Success) {
        return ClientError::SendMessageFailed;
    }

    // (3-2) recv welcome reply msg
    ret = Communicator::recvMessage(handle.sockFD, msgHdr, handle.buffer, false);
    if (ret != Communicator::Success)
        return ClientError::RecvMessageFailed;

    SASSERT0(msgHdr.getMsgType() == MessageHeader::WelcomeReply);

    handle.hasSession = true;

    return ClientError::Success;
}

ClientError ClientAPI::releaseSession(ClientHandle handle) {

    ClientError retValue = ClientError::Success;

    if (!handle.hasSession)
        return ClientError::NoSession;

    MessageHeader msgHdr;
    msgHdr.setMsgType(MessageHeader::GoodBye);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    MsgSerializer::serializeMsgHdr(msgHdr, handle.buffer);

    Communicator::CommRetType ret;
    ret = Communicator::sendMessage(handle.sockFD, msgHdr, handle.buffer);
    if (ret != Communicator::Success) {
        retValue = ClientError::SendMessageFailed;
    }

    close(handle.sockFD);
    handle.hasSession = false;

    if (handle.buffer != NULL)
        free(handle.buffer);

    return retValue;
}

ClientError ClientAPI::createNetwork(ClientHandle handle, std::string networkDef,
    NetworkHandle& netHandle) {

    if (!handle.hasSession)
        return ClientError::NoSession;

    Job* createNetworkJob = new Job(JobType::CreateNetwork);
    createNetworkJob->addJobElem(Job::StringType, strlen(networkDef.c_str()),
        (void*)networkDef.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(createNetworkJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            createNetworkJob);
    delete createNetworkJob;
    CPPAPI_CHECK_SEND();

    Job* createNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &createNetworkReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(createNetworkReplyJob, bufSize);

    SASSERT0(createNetworkReplyJob->getType() == JobType::CreateNetworkReply);
    string networkID = createNetworkReplyJob->getStringValue(0);
    delete createNetworkReplyJob;

    netHandle.networkID = networkID;
    netHandle.created = true;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::createNetworkFromFile(ClientHandle handle,
    std::string filePathInServer, NetworkHandle& netHandle) {

    if (!handle.hasSession)
        return ClientError::NoSession;

    Job* createNetworkFromFileJob = new Job(JobType::CreateNetworkFromFile);
    createNetworkFromFileJob->addJobElem(Job::StringType, strlen(filePathInServer.c_str()),
        (void*)filePathInServer.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(createNetworkFromFileJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            createNetworkFromFileJob);
    delete createNetworkFromFileJob;
    CPPAPI_CHECK_SEND();

    Job* createNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &createNetworkReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(createNetworkReplyJob, bufSize);

    SASSERT0(createNetworkReplyJob->getType() == JobType::CreateNetworkReply);
    string networkID = createNetworkReplyJob->getStringValue(0);
    delete createNetworkReplyJob;

    netHandle.networkID = networkID;
    netHandle.created = true;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::createResumeNetwork(ClientHandle handle, std::string networkID,
        int keepHistory, NetworkHandle& netHandle) {

    if (!handle.hasSession)
        return ClientError::NoSession;

    Job* createResumeNetworkJob = new Job(JobType::CreateResumeNetwork);
    createResumeNetworkJob->addJobElem(Job::StringType, 
            strlen(networkID.c_str()), (void*)networkID.c_str());
    createResumeNetworkJob->addJobElem(Job::IntType, 1, (void*)&keepHistory);

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(createResumeNetworkJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            createResumeNetworkJob);
    delete createResumeNetworkJob;
    CPPAPI_CHECK_SEND();

    Job* createResumeNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &createResumeNetworkReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(createResumeNetworkReplyJob, bufSize);

    SASSERT0(createResumeNetworkReplyJob->getType() == JobType::CreateResumeNetworkReply);
    string newNetworkID = createResumeNetworkReplyJob->getStringValue(0);
    delete createResumeNetworkReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    if (newNetworkID == "")
        return ClientError::CreateResumeNetworkFailed;

    netHandle.networkID = newNetworkID;
    netHandle.created = true;

    return ClientError::Success;
}

ClientError ClientAPI::stopNetwork(ClientHandle handle, std::string networkID) {

    if (!handle.hasSession)
        return ClientError::NoSession;

    Job* stopNetworkJob = new Job(JobType::StopNetwork);
    stopNetworkJob->addJobElem(Job::StringType, 
            strlen(networkID.c_str()), (void*)networkID.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(stopNetworkJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            stopNetworkJob);
    delete stopNetworkJob;
    CPPAPI_CHECK_SEND();

    Job* stopNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &stopNetworkReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(stopNetworkReplyJob, bufSize);

    SASSERT0(stopNetworkReplyJob->getType() == JobType::StopNetworkReply);
    string newNetworkID = stopNetworkReplyJob->getStringValue(0);
    delete stopNetworkReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::destroyNetwork(ClientHandle handle, NetworkHandle& netHandle) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* destroyNetworkJob = new Job(JobType::DestroyNetwork);
    destroyNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
            (void*)netHandle.networkID.c_str());
    int ret = Client::sendJob(handle.sockFD, handle.buffer, destroyNetworkJob);
    delete destroyNetworkJob;
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* destroyNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, handle.buffer, &destroyNetworkReplyJob, &bufSize);
    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;
    SASSERT0(destroyNetworkReplyJob->getType() == JobType::DestroyNetworkReply);
    delete destroyNetworkReplyJob;

    netHandle.created = false;
    return ClientError::Success;
}

ClientError ClientAPI::buildNetwork(ClientHandle handle, NetworkHandle netHandle,
    int epochs) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* buildNetworkJob = new Job(JobType::BuildNetwork);
    buildNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    buildNetworkJob->addJobElem(Job::IntType, 1, (void*)&epochs);
    int ret = Client::sendJob(handle.sockFD, handle.buffer, buildNetworkJob);
    delete buildNetworkJob;

    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* buildNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, handle.buffer, &buildNetworkReplyJob, &bufSize);

    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;
    SASSERT0(buildNetworkReplyJob->getType() == JobType::BuildNetworkReply);
    delete buildNetworkReplyJob;

    return ClientError::Success;
}

ClientError ClientAPI::resetNetwork(ClientHandle handle, NetworkHandle netHandle) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* resetNetworkJob = new Job(JobType::ResetNetwork);
    resetNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    int ret = Client::sendJob(handle.sockFD, handle.buffer, resetNetworkJob);
    delete resetNetworkJob;
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* resetNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, handle.buffer, &resetNetworkReplyJob, &bufSize);
    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;
    SASSERT0(resetNetworkReplyJob->getType() == JobType::ResetNetworkReply);
    delete resetNetworkReplyJob;

    return ClientError::Success;
}

ClientError ClientAPI::runNetwork(ClientHandle handle, NetworkHandle netHandle, 
    bool inference) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* runNetworkJob = new Job(JobType::RunNetwork);
    runNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    int inferenceInt = (int)inference;
    runNetworkJob->addJobElem(Job::IntType, 1, (void*)&inferenceInt);
    int ret = Client::sendJob(handle.sockFD, handle.buffer, runNetworkJob);
    delete runNetworkJob;
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* runNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, handle.buffer, &runNetworkReplyJob, &bufSize);
    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;
    SASSERT0(runNetworkReplyJob->getType() == JobType::RunNetworkReply);
    int success = runNetworkReplyJob->getIntValue(0);
    delete runNetworkReplyJob;

    if (success == 1)
        return ClientError::Success;
    else
        return ClientError::RunNetworkFailed;
}

ClientError ClientAPI::runNetworkMiniBatch(ClientHandle handle, NetworkHandle netHandle,
    bool inference, int miniBatchIdx) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* runNetworkJob = new Job(JobType::RunNetworkMiniBatch);
    runNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    int inferenceInt = (int)inference;
    runNetworkJob->addJobElem(Job::IntType, 1, (void*)&inferenceInt);
    runNetworkJob->addJobElem(Job::IntType, 1, (void*)&miniBatchIdx);
    int ret = Client::sendJob(handle.sockFD, handle.buffer, runNetworkJob);
    delete runNetworkJob;
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* runNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, handle.buffer, &runNetworkReplyJob, &bufSize);
    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;
    SASSERT0(runNetworkReplyJob->getType() == JobType::RunNetworkReply);
    int success = runNetworkReplyJob->getIntValue(0);
    delete runNetworkReplyJob;

    if (success == 1)
        return ClientError::Success;
    else
        return ClientError::RunNetworkFailed;
}

ClientError ClientAPI::saveNetwork(ClientHandle handle, NetworkHandle netHandle,
    std::string filePath) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* saveNetworkJob = new Job(JobType::SaveNetwork);
    saveNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    saveNetworkJob->addJobElem(Job::StringType, strlen(filePath.c_str()),
        (void*)filePath.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(saveNetworkJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            saveNetworkJob);
    delete saveNetworkJob;
    CPPAPI_CHECK_SEND();

    Job* saveNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &saveNetworkReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(saveNetworkReplyJob, bufSize);

    SASSERT0(saveNetworkReplyJob->getType() == JobType::SaveNetworkReply);
    delete saveNetworkReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::loadNetwork(ClientHandle handle, NetworkHandle netHandle,
    std::string filePath) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* loadNetworkJob = new Job(JobType::LoadNetwork);
    loadNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    loadNetworkJob->addJobElem(Job::StringType, strlen(filePath.c_str()),
        (void*)filePath.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(loadNetworkJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            loadNetworkJob);
    delete loadNetworkJob;
    CPPAPI_CHECK_SEND();

    Job* loadNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &loadNetworkReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(loadNetworkReplyJob, bufSize);

    SASSERT0(loadNetworkReplyJob->getType() == JobType::LoadNetworkReply);
    delete loadNetworkReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::getObjectDetection(ClientHandle handle, NetworkHandle netHandle,
    int channel, int height, int width, float* imageData, vector<BoundingBox>& boxArray,
    int coordRelative) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* runJob = new Job(JobType::RunNetworkWithInputData);
    runJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    runJob->addJobElem(Job::IntType, 1, (void*)&channel);
    runJob->addJobElem(Job::IntType, 1, (void*)&height);
    runJob->addJobElem(Job::IntType, 1, (void*)&width);
    runJob->addJobElem(Job::IntType, 1, (void*)&coordRelative);

    int imageDataElemCount = channel * height * width;
    runJob->addJobElem(Job::FloatArrayType, imageDataElemCount, imageData);

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
        &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::RunNetworkWithInputDataReply);
    
    int resultBoxCount = runReplyJob->getIntValue(0);
    int elemIdx = 1;
    for (int i = 0; i < resultBoxCount; i++) {
        BoundingBox bbox;
        bbox.top        = runReplyJob->getFloatValue(elemIdx + 0);
        bbox.left       = runReplyJob->getFloatValue(elemIdx + 1);
        bbox.bottom     = runReplyJob->getFloatValue(elemIdx + 2);
        bbox.right      = runReplyJob->getFloatValue(elemIdx + 3);
        bbox.confidence = runReplyJob->getFloatValue(elemIdx + 4);
        bbox.labelIndex = runReplyJob->getIntValue(elemIdx + 5);

        boxArray.push_back(bbox);
        elemIdx += 6;
    }
    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::runObjectDetectionWithInput(ClientHandle handle,
    NetworkHandle netHandle, int channel, int height, int width, float* imageData,
    vector<BoundingBox>& boxArray, int baseNetworkType, int needRecovery) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* runJob = new Job(JobType::RunObjectDetectionNetworkWithInput);
    runJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    runJob->addJobElem(Job::IntType, 1, (void*)&channel);
    runJob->addJobElem(Job::IntType, 1, (void*)&height);
    runJob->addJobElem(Job::IntType, 1, (void*)&width);
    runJob->addJobElem(Job::IntType, 1, (void*)&baseNetworkType);
    runJob->addJobElem(Job::IntType, 1, (void*)&needRecovery);

    int imageDataElemCount = channel * height * width;
    runJob->addJobElem(Job::FloatArrayType, imageDataElemCount, imageData);

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::RunObjectDetectionNetworkWithInputReply);
    
    int resultBoxCount = runReplyJob->getIntValue(0);

    if (needRecovery && (resultBoxCount == -1)) {
        delete runReplyJob;

        if (bigBuffer != NULL)
            free(bigBuffer);

        return ClientError::RunAdhocNetworkFailed;
    }

    int elemIdx = 1;
    for (int i = 0; i < resultBoxCount; i++) {
        BoundingBox bbox;
        bbox.top        = runReplyJob->getFloatValue(elemIdx + 0);
        bbox.left       = runReplyJob->getFloatValue(elemIdx + 1);
        bbox.bottom     = runReplyJob->getFloatValue(elemIdx + 2);
        bbox.right      = runReplyJob->getFloatValue(elemIdx + 3);
        bbox.confidence = runReplyJob->getFloatValue(elemIdx + 4);
        bbox.labelIndex = runReplyJob->getIntValue(elemIdx + 5);

        boxArray.push_back(bbox);
        elemIdx += 6;
    }
    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::runClassificationWithInput(ClientHandle handle,
    NetworkHandle netHandle, int channel, int height, int width, float* imageData,
    vector<pair<int, float>>& labelIndexArray, int baseNetworkType, int maxResultCount, 
    int needRecovery) {

    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* runJob = new Job(JobType::RunClassificationNetworkWithInput);
    runJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    runJob->addJobElem(Job::IntType, 1, (void*)&channel);
    runJob->addJobElem(Job::IntType, 1, (void*)&height);
    runJob->addJobElem(Job::IntType, 1, (void*)&width);
    runJob->addJobElem(Job::IntType, 1, (void*)&baseNetworkType);
    runJob->addJobElem(Job::IntType, 1, (void*)&maxResultCount);
    runJob->addJobElem(Job::IntType, 1, (void*)&needRecovery);

    int imageDataElemCount = channel * height * width;
    runJob->addJobElem(Job::FloatArrayType, imageDataElemCount, imageData);

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::RunClassificationNetworkWithInputReply);
    
    int resultCount = runReplyJob->getIntValue(0);
    for (int i = 0; i < resultCount; i++) {
        int labelIndex = runReplyJob->getIntValue(2 * i + 1);
        float score = runReplyJob->getFloatValue(2 * i + 2);

        labelIndexArray.push_back(make_pair(labelIndex, score));
    }
    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    if (needRecovery && (resultCount == -1))
        ClientError::RunAdhocNetworkFailed;

    return ClientError::Success;
}

ClientError ClientAPI::getMeasureItemName(ClientHandle handle, string networkID,
    vector<string>& measureItemNames) {

    Job* runJob = new Job(JobType::GetMeasureItemName);
    runJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::GetMeasureItemNameReply);

    int resultItemCount = runReplyJob->getIntValue(0);
    for (int i = 0; i < resultItemCount; i++) {
        string itemName = runReplyJob->getStringValue(i + 1);
        measureItemNames.push_back(itemName);
    }
    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    if (resultItemCount == -1)
        return ClientError::RequestedNetworkNotExist;

    return ClientError::Success;
}

ClientError ClientAPI::getMeasures(ClientHandle handle, string networkID,
    bool forwardSearch, int start, int count, int* startIterNum, int* dataCount,
    int* curIterNum, int* totalIterNum, float* data) {

    int forward = (int)forwardSearch;

    Job* runJob = new Job(JobType::GetMeasures);
    runJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());
    runJob->addJobElem(Job::IntType, 1, (void*)&forward);
    runJob->addJobElem(Job::IntType, 1, (void*)&start);
    runJob->addJobElem(Job::IntType, 1, (void*)&count);

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::GetMeasuresReply);

    int measureCount = runReplyJob->getIntValue(0);
    (*dataCount) = measureCount;
    (*startIterNum) = runReplyJob->getIntValue(1);
    (*curIterNum) = runReplyJob->getIntValue(2);
    (*totalIterNum) = runReplyJob->getIntValue(3);

    if ((*dataCount) > 0) {
        float *measures = runReplyJob->getFloatArray(4);
        memcpy(data, measures, sizeof(float) * measureCount);
    }

    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    if (measureCount == -1)
        return ClientError::RequestedNetworkNotExist;

    return ClientError::Success;
}

ClientError ClientAPI::getNetworkEvent(ClientHandle handle, string networkID,
        vector<NetEvent> &events) {

    Job* runJob = new Job(JobType::GetNetworkEvent);
    runJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::GetNetworkEventReply);

    int eventCount = runReplyJob->getIntValue(0); 
    for (int i = 0; i < eventCount; i++) {
        NetEvent event;
        event.eventType     = runReplyJob->getIntValue(i * 4 + 1);

        strncpy(event.eventTime, runReplyJob->getStringValue(i * 4 + 2).c_str(),
                NETEVENT_EVENTTIME_STRING_LENGTH - 1);
        event.eventTime[NETEVENT_EVENTTIME_STRING_LENGTH - 1] = '\0';

        event.layerID       = runReplyJob->getIntValue(i * 4 + 3);

        string msg = runReplyJob->getStringValue(i * 4 + 4);
        int copyLen = min(NETEVENT_MESSAGE_STRING_LENGTH - 1, (int)strlen(msg.c_str()));
        strncpy(event.message, msg.c_str(), copyLen);
        event.message[copyLen] = '\0';

        events.push_back(event);
    }

    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::getNetworkEventMsg(ClientHandle handle, string networkID,
        vector<string> &msgs) {

    Job* runJob = new Job(JobType::GetNetworkEventMessage);
    runJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::GetNetworkEventMessageReply);

    int eventCount = runReplyJob->getIntValue(0); 
    for (int i = 0; i < eventCount; i++) {
        string msg = runReplyJob->getStringValue(i + 1);
        msgs.push_back(msg);
    }

    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::checkNetworkDef(ClientHandle handle, string networkDef,
        int &resultCode, int &gpuMBSize, int &layerID, string &errorMsg) {

    Job* runJob = new Job(JobType::CheckNetworkDef);
    runJob->addJobElem(Job::StringType, strlen(networkDef.c_str()),
            (void*)networkDef.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::CheckNetworkDefReply);

    resultCode = runReplyJob->getIntValue(0);
    gpuMBSize = runReplyJob->getIntValue(1);
    layerID = runReplyJob->getIntValue(2);
    errorMsg = runReplyJob->getStringValue(3);

    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    if (resultCode == 1)
        return ClientError::NetworkValidationFailed;
    else if (resultCode == 2)
        return ClientError::InsufficientGPUMem;
    else
        return ClientError::Success;
}


ClientError ClientAPI::getNetworkProgress(ClientHandle handle, std::string networkID,
        int &curIterNum, int &totalIterNum) {

    Job* runJob = new Job(JobType::GetNetworkProgress);
    runJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::GetNetworkProgressReply);

    curIterNum = runReplyJob->getIntValue(0);
    totalIterNum = runReplyJob->getIntValue(1);

    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::getNetworkResult(ClientHandle handle, std::string networkID,
        vector<string> &itemNames, vector<float> &itemResults) {

    Job* runJob = new Job(JobType::GetNetworkResult);
    runJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::GetNetworkResultReply);

    int itemCount = runReplyJob->getIntValue(0);
    for (int i = 0; i < itemCount; i++) {
        string itemName = runReplyJob->getStringValue(i * 2 + 1);
        itemNames.push_back(itemName);
        float itemResult = runReplyJob->getFloatValue(i * 2 + 2);
        itemResults.push_back(itemResult);
    }

    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}
