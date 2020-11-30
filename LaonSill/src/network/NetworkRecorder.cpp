/**
 * @file NetworkRecorder.cpp
 * @date 2018-03-30
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <assert.h>

#include "NetworkRecorder.h"
#include "WorkContext.h"

using namespace std;

map<string, NetworkEventRecord*>    NetworkRecorder::records;
mutex                               NetworkRecorder::recordMutex;

std::string fmtLogToString(const char * fmt, ...) {
    char buf[FMTTOSTRING_BUFSIZE];
    va_list ap;

    va_start(ap, fmt);
    int needed = vsnprintf(buf, FMTTOSTRING_BUFSIZE, fmt, ap);
    va_end(ap);

    return std::string(buf, std::min(FMTTOSTRING_BUFSIZE, needed));
}

NetworkRecorder::NetworkRecorder() {

}

NetworkRecorder::~NetworkRecorder() {

}

void NetworkRecorder::pushEvent(NetworkEventType eventType, string msg) {
    string networkID = WorkContext::curNetworkID;
    int layerID = WorkContext::curLayerID;
    NetworkEventRecord *record;
    unique_lock<mutex> recordLock(NetworkRecorder::recordMutex);
    if (NetworkRecorder::records.find(networkID) == NetworkRecorder::records.end()) {
        // FIXME: 지금은 push한 Event를 딱히 반납하지 않고 있다.
        //        추후에 적정 개수 이상이면 오래된 애들은 반납하도록 수정할 예정이다.
        //        수정 전까지는 SNEW를 쓰지 않고, new를 사용하도록 하겠다.
        //        SNEW를 쓰게 되면 memory leak으로 착각할 수 있기 때문이다.
        NetworkEventRecord *newRecord = new NetworkEventRecord();
        NetworkRecorder::records[networkID] = newRecord;
    }
    record = NetworkRecorder::records[networkID];
    recordLock.unlock();

    NetworkEvent event;
    event.eventType = eventType;
    event.eventTime = time(NULL);
    event.msg = msg;
    event.layerID = layerID;
    unique_lock<mutex> eventLock(record->eventMutex);
    record->events.push_back(event);
}

void NetworkRecorder::getEvents(string networkID, vector<NetworkEvent> &events) {
    NetworkEventRecord *record;
    unique_lock<mutex> recordLock(NetworkRecorder::recordMutex);
    if (NetworkRecorder::records.find(networkID) == NetworkRecorder::records.end()) {
        return;
    }
    record = NetworkRecorder::records[networkID];
    recordLock.unlock();
    
    unique_lock<mutex> eventLock(record->eventMutex);
    for (int i = 0; i < record->events.size(); i++) {
        events.push_back(record->events[i]);
    }
}

void NetworkRecorder::getEventMsgs(string networkID, vector<string> &eventMsgs) {
    NetworkEventRecord *record;
    unique_lock<mutex> recordLock(NetworkRecorder::recordMutex);
    if (NetworkRecorder::records.find(networkID) == NetworkRecorder::records.end()) {
        return;
    }
    record = NetworkRecorder::records[networkID];
    recordLock.unlock();
    
    unique_lock<mutex> eventLock(record->eventMutex);
    for (int i = 0; i < record->events.size(); i++) {
        string eventStr;
        eventStr = timeToString(record->events[i].eventTime) + " " + 
            typeToString(record->events[i].eventType) + " " + record->events[i].msg;
        eventMsgs.push_back(eventStr);
    }
}

string NetworkRecorder::timeToString(time_t t) {
    char s[20];
    sprintf(s, "%04d-%02d-%02d %02d:%02d:%02d", localtime(&t)->tm_year + 1900,
            localtime(&t)->tm_mon + 1, localtime(&t)->tm_mday, localtime(&t)->tm_hour,
            localtime(&t)->tm_min, localtime(&t)->tm_sec);
    return string(s);
}

string NetworkRecorder::typeToString(NetworkEventType eventType) {
    switch (eventType) {
        case NETWORK_EVENT_TYPE_eDEBUG:
            return "[DEBUG]";

        case NETWORK_EVENT_TYPE_eWARNING:
            return "[WARNING]";

        case NETWORK_EVENT_TYPE_eINFO:
            return "[INFO]";

        case NETWORK_EVENT_TYPE_eERROR:
            return "[ERROR]";

        case NETWORK_EVENT_TYPE_eSYSTEM:
            return "[SYSTEM]";

        case NETWORK_EVENT_TYPE_eASSERT:
            return "[ASSERT]";

        case NETWORK_EVENT_TYPE_eVALIDATION:
            return "[VALIDATION]";

        default:
            assert(false);
    }

    // should not be reached...
    return "[?]";
}

bool NetworkRecorder::getValidationEvent(string networkID, NetworkEvent &event) {
    vector<NetworkEvent> netEvents;
    NetworkRecorder::getEvents(networkID, netEvents);

    bool found = false;

    for (int i = 0; i < netEvents.size(); i++) {
        if (netEvents[i].eventType == NETWORK_EVENT_TYPE_eVALIDATION) {
            event = netEvents[i];
            found = true;
            break;
        }
    }

    return found;
}
