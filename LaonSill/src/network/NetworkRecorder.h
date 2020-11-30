/**
 * @file NetworkRecorder.h
 * @date 2018-03-30
 * @author moonhoen lee
 * @brief 네트워크의 상태정보(오류정보 포함)를 관리하는 모듈
 * @details
 */

#ifndef NETWORKRECORDER_H
#define NETWORKRECORDER_H

#include <stdarg.h>
#include <string.h>

#include <map>
#include <vector>
#include <mutex>
#include <string>
#include <ctime>

#define FMTTOSTRING_BUFSIZE     2048

std::string fmtLogToString(const char * fmt, ...);

#define SEVENT_PUSH(eventType, fmt, args...)                                            \
    do {                                                                                \
        NetworkRecorder::pushEvent(eventType, fmtLogToString(fmt, ##args));             \
    } while (0)

#define SEVENT_PUSH0(eventType)                                                         \
    do {                                                                                \
        NetworkRecorder::pushEvent(eventType, "");                                      \
    } while (0)

// DEBUG, WARNING, INFO, ERROR는 반드시 ColdLog.h의 LogLevel과 동일해야 한다.
typedef enum NetworkEventType_e {
    NETWORK_EVENT_TYPE_eDEBUG = 0,
    NETWORK_EVENT_TYPE_eWARNING = 1,
    NETWORK_EVENT_TYPE_eINFO = 2,
    NETWORK_EVENT_TYPE_eERROR = 3,
    NETWORK_EVENT_TYPE_eSYSTEM,
    NETWORK_EVENT_TYPE_eASSERT,
    NETWORK_EVENT_TYPE_eVALIDATION,
    NETWORK_EVENT_TYPE_eMAX
} NetworkEventType;

typedef struct NetworkEvent_s {
    NetworkEventType    eventType;
    std::time_t         eventTime;
    int                 layerID;
    std::string         msg;
} NetworkEvent;

typedef struct NetworkEventRecord_s {
    std::mutex                  eventMutex;
    std::vector<NetworkEvent>   events;
} NetworkEventRecord;

class NetworkRecorder {
public: 
    NetworkRecorder();
    virtual ~NetworkRecorder();

    static void pushEvent(NetworkEventType eventType, std::string msg);
    static void getEvents(std::string networkID, std::vector<NetworkEvent> &events);
    static void getEventMsgs(std::string networkID, std::vector<std::string> &eventMsgs);
    static bool getValidationEvent(std::string networkID, NetworkEvent &event);
    static std::string  timeToString(std::time_t t);
private:
    static std::string  typeToString(NetworkEventType eventType);
    static std::map<std::string, NetworkEventRecord*> records;
    static std::mutex   recordMutex;
};

#endif /* NETWORKRECORDER_H */
