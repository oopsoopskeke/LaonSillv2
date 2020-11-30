/**
 * @file Client.cpp
 * @date 2016-10-25
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

#include "Client.h"
#include "Communicator.h"
#include "MsgSerializer.h"
#include "MessageHeader.h"
#include "SysLog.h"

using namespace std;

const int MAX_RETRY_SECOND = 128;
int Client::connectRetry(int sockFd, const struct sockaddr *sockAddr, socklen_t sockLen) {
    int nsec;

    for (nsec = 1; nsec <= MAX_RETRY_SECOND; nsec <<= 1) {
        if (connect(sockFd, sockAddr, sockLen) == 0) {
            return 0;
        }

        if (nsec <= MAX_RETRY_SECOND/2)
            sleep(nsec);
    }

    return -1;
}

int Client::sendJob(int fd, char* buf, Job* job) {
    // see handlePushJobMsg()@Communicator.cpp && JobType enumeration @ Job.h
    // (1) send job msg
    MessageHeader msgHdr;
    Communicator::sendJobToBuffer(msgHdr, job, buf);
    Communicator::CommRetType ret = Communicator::sendMessage(fd, msgHdr, buf);
    if (ret != Communicator::Success)
        return (int)ret;

    // (2) recv msg
    ret = Communicator::recvMessage(fd, msgHdr, buf, false);
    if (ret != Communicator::Success)
        return (int)ret;

    SASSERT0(msgHdr.getMsgType() == MessageHeader::PushJobReply);
    return (int)ret;
}

int Client::recvJob(int fd, char* buf, Job** job, int* bufLen) {
    // see handlePushJobMsg()@Communicator.cpp && JobType enumeration @ Job.h
    MessageHeader msgHdr;

    Communicator::CommRetType ret;
    if ((*bufLen) > MessageHeader::MESSAGE_DEFAULT_SIZE)
        ret = Communicator::recvMessage(fd, msgHdr, buf, true);
    else
        ret = Communicator::recvMessage(fd, msgHdr, buf, false);

    if (ret != Communicator::Success) {
        if (ret == Communicator::RecvOnlyHeader) {
            (*bufLen) = msgHdr.getMsgLen();
        } else {
            (*bufLen) = -1;
        }
        return (int)ret;
    }

    SASSERT0(msgHdr.getMsgType() == MessageHeader::PushJob);
    Communicator::recvJobFromBuffer(job, buf);
    (*bufLen) = msgHdr.getMsgLen();

    return (int)ret;
}

void Client::clientMain(const char* hostname, int portno) {
    int     sockFd, err;

    // (1) get server info (struct hostent)
    struct hostent *server;
    server = gethostbyname(hostname);
    if (server == NULL) {
        printf("ERROR: no such host as %s\n", hostname);
        exit(0);
    }

    // (2) create socket & connect to the server
    sockFd = socket(AF_INET, SOCK_STREAM, 0);
    SASSERT0(sockFd != -1);
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(struct sockaddr_in));
    serverAddr.sin_family = AF_INET;
    memcpy((char*)&serverAddr.sin_addr.s_addr, (char*)server->h_addr, server->h_length);
    serverAddr.sin_port = htons(portno);

    int connectRetry = Client::connectRetry(sockFd, (struct sockaddr*)&serverAddr,
                                            sizeof(serverAddr));
    if (connectRetry == -1) {
        printf("ERROR: connect failed\n");
        exit(0);
    }

    // (3-1) send welcome msg
    cout << "send welcome msg" << endl;
    char* buf = (char*)malloc(MessageHeader::MESSAGE_DEFAULT_SIZE);
    SASSERT0(buf != NULL);

    MessageHeader msgHdr;
    msgHdr.setMsgType(MessageHeader::Welcome);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    MsgSerializer::serializeMsgHdr(msgHdr, buf);
    Communicator::CommRetType ret = Communicator::sendMessage(sockFd, msgHdr, buf);
    SASSERT0(ret == Communicator::Success);

    // (3-2) recv welcome reply msg
    cout << "recv welcome msg" << endl;
    ret = Communicator::recvMessage(sockFd, msgHdr, buf, false);
    SASSERT0(ret == Communicator::Success);
    SASSERT0(msgHdr.getMsgType() == MessageHeader::WelcomeReply);

    // (4) create network
    cout << "send create-network job" << endl;
    Job* createNetworkJob = new Job(JobType::CreateNetworkFromFile);
    string networkFilePath = "network.conf.test";
    createNetworkJob->addJobElem(Job::StringType, strlen(networkFilePath.c_str()),
        (void*)networkFilePath.c_str());
    int retval = Client::sendJob(sockFd, buf, createNetworkJob);
    delete createNetworkJob;
    SASSERT0(retval == Communicator::Success);

    Job* createNetworkReplyJob;
    int bufSize = 0;
    retval = recvJob(sockFd, buf, &createNetworkReplyJob, &bufSize);
    SASSERT0(retval == Communicator::Success);
    SASSERT0(createNetworkReplyJob->getType() == JobType::CreateNetworkReply);
    string networkID = createNetworkReplyJob->getStringValue(0);
    cout << "created network ID : " << networkID << endl;
    delete createNetworkReplyJob;

#if 0
    // (6) send Halt Msg
    cout << "send halt msg" << endl;
    msgHdr.setMsgType(MessageHeader::HaltMachine);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    MsgSerializer::serializeMsgHdr(msgHdr, buf);
    ret = Communicator::sendMessage(sockFd, msgHdr, buf);
    SASSERT0(ret == Communicator::Success);
#else
    // (6) send goodbye msg
    cout << "send goodbye msg" << endl;
    msgHdr.setMsgType(MessageHeader::GoodBye);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    MsgSerializer::serializeMsgHdr(msgHdr, buf);
    ret = Communicator::sendMessage(sockFd, msgHdr, buf);
    SASSERT0(ret == Communicator::Success);
#endif

    // XXX: process should wait until send buffer is empty
    // cleanup resrouce & exit
    close(sockFd);
    free(buf);
}
