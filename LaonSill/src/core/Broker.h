/**
 * @file Broker.h
 * @date 2016-12-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef BROKER_H
#define BROKER_H 

#include <map>
#include <list>
#include <mutex>
#include <condition_variable>

#include "Job.h"

#define MAKE_JOBTASK_KEY(jobID, taskID)     ((uint64_t)jobID << 32 | (uint64_t)taskID)

class Broker {

public: 
    enum BrokerRetType : int {
        Success = 0,
        NoContent,
        NoFreeAdhocSess,
        BrokerRetTypeMax
    };

    enum BrokerAccessType : int {
        Blocking = 0,
        NoBlocking,
        BrokerAccessTypeMax
    };

    struct AdhocSess {
        int                         arrayIdx;
        int                         sessID;     // SPARAM(BROKER_ADHOC_SESS_STARTS) 이상
        std::mutex                 *sessMutex;
        std::condition_variable    *sessCondVar;
        bool                        found;
    };
                                    Broker() {}
    virtual                        ~Broker() {}

    static BrokerRetType            publish(int jobID, Job *content);
    static BrokerRetType            publishEx(int jobID, int taskID, Job *content);
    static BrokerRetType            subscribe(int jobID, Job **content,
                                        BrokerAccessType access);
    static BrokerRetType            subscribeEx(int jobID, int taskID, Job **content,
                                        BrokerAccessType access);
    static void                     init();
    static void                     destroy();

private:
    static std::map<uint64_t, Job*> contentMap;     // published map
    static std::mutex               contentMutex;
    static std::map<uint64_t, int>  waiterMap;      // waiting subscriber map
    static std::mutex               waiterMutex;

    // adhocSessArray의 인덱스에 해당하는 AdhocSess의 sessID와의 관계는
    // index = sessID - SPARAM(BROKER_ADHOC_SESS_STARTS) 이어야 한다.
    static AdhocSess               *adhocSessArray; 
    
    static std::list<int>           freeAdhocSessIDList;
    static std::mutex               adhocSessMutex; 
    static int                      getFreeAdhocSessID();
    static void                     releaseAdhocSessID(int sessID);
};

#endif /* BROKER_H */
