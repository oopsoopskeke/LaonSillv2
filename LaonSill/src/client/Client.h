/**
 * @file Client.h
 * @date 2016-10-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef CLIENT_H
#define CLIENT_H 

#include <sys/socket.h>

#include "common.h"
#include "Job.h"

class Client {
public:
                        Client() {}
    virtual            ~Client() {}
    static void         clientMain(const char* hostname, int portno);

    static int          sendJob(int fd, char* buf, Job* job);
    static int          recvJob(int fd, char* buf, Job** job, int* bufLen);
    static int          connectRetry(int socketFd, const struct sockaddr *sockAddr,
                            socklen_t sockLen);
};

#endif /* CLIENT_H */
