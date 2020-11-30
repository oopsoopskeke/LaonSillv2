/**
 * @file ClientAPI.h
 * @date 2017-06-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef CLIENTAPI_H
#define CLIENTAPI_H 

#include <string>
#include <vector>
#include <utility>

#define MAX_SERVER_HOSTNAME_LEN             (1024)

typedef struct ClientHandle_s {
    int     sockFD;
    char    serverHostName[MAX_SERVER_HOSTNAME_LEN];
    int     serverPortNum;
    char*   buffer;
    int     bufLen;
    bool    hasSession;

    ClientHandle_s() {
        buffer = NULL;
        bufLen = 0;
        hasSession = false;
    }
} ClientHandle;

typedef enum ClientError_s {
    Success = 0,
    TooLongServerHostName,
    NoSuchHost,
    HostConnectionFailed,
    ClientHandleBufferAllocationFailed,
    ClientHandleBufferReallocationFailed,
    ClientHandleInternalMemoryAllocationFailed,
    SendMessageFailed,
    RecvMessageFailed,
    HaveSessionAlready,
    NoSession,
    NotCreatedNetwork,
    RunNetworkFailed,
    RequestedNetworkNotExist,
    SendJobFailed,
    RecvJobFailed,
    RunAdhocNetworkFailed,
    CreateResumeNetworkFailed,
    InsufficientGPUMem,
    NetworkValidationFailed,
    SessErrorMax
} ClientError;

typedef struct NetworkHandle_s {
    std::string     networkID;
    bool            created;

    NetworkHandle_s() {
        networkID = "";
        created = false;
    }
} NetworkHandle;

typedef struct BoundingBox_s {
    float top;
    float left;
    float bottom;
    float right;
    float confidence;
    int   labelIndex;
} BoundingBox;

#define NETEVENT_EVENTTIME_STRING_LENGTH        (20)
#define NETEVENT_MESSAGE_STRING_LENGTH          (2048)

typedef struct NetEvent_s {
    int     eventType;
    char    eventTime[NETEVENT_EVENTTIME_STRING_LENGTH];
    int     layerID;
    char    message[NETEVENT_MESSAGE_STRING_LENGTH];
} NetEvent;

class ClientAPI {
public: 
                        ClientAPI() {}
    virtual            ~ClientAPI() {}

    static ClientError      createHandle(ClientHandle& handle, std::string serverHostName,
                                         int serverPortNum);


    static ClientError      getSession(ClientHandle& handle);
    static ClientError      releaseSession(ClientHandle handle);

    static ClientError      createNetwork(ClientHandle handle, std::string networkDef,
                                          NetworkHandle& netHandle);
    static ClientError      createNetworkFromFile(ClientHandle handle,
                                                  std::string filePathInServer,
                                                  NetworkHandle& netHandle);
    static ClientError      createResumeNetwork(ClientHandle handle, std::string networkID,
                                                int keepHistory, NetworkHandle& netHandle);
    static ClientError      stopNetwork(ClientHandle handle, std::string networkID);
    static ClientError      destroyNetwork(ClientHandle handle, NetworkHandle& netHandle);
    static ClientError      buildNetwork(ClientHandle handle, NetworkHandle netHandle,
                                         int epochs);
    static ClientError      resetNetwork(ClientHandle handle, NetworkHandle netHandle);
    static ClientError      runNetwork(ClientHandle handle, NetworkHandle netHandle,
                                       bool inference);
    static ClientError      runNetworkMiniBatch(ClientHandle handle, NetworkHandle netHandle,
                                                bool inference, int miniBatchIdx);
    static ClientError      saveNetwork(ClientHandle handle, NetworkHandle netHandle,
                                        std::string filePath);
    static ClientError      loadNetwork(ClientHandle handle, NetworkHandle netHandle,
                                        std::string filePath);

    static ClientError      getObjectDetection(ClientHandle handle, NetworkHandle netHandle,
                                int channel, int height, int width, float* imageData,
                                std::vector<BoundingBox>& boxArray, int coordRelative=0);

    static ClientError      runObjectDetectionWithInput(ClientHandle handle,
                                NetworkHandle netHandle, int channel, int height,
                                int width, float* imageData, 
                                std::vector<BoundingBox>& boxArray, int baseNetworkType, 
                                int needRecovery);

    static ClientError      runClassificationWithInput(ClientHandle handle,
                                NetworkHandle netHandle, int channel, int height,
                                int width, float* imageData, 
                                std::vector<std::pair<int, float>>& labelIndexArray, 
                                int baseNetworkType, int maxResultCount, int needRecovery);

    static ClientError      getMeasureItemName(ClientHandle handle, std::string networkID,
                                std::vector<std::string>& measureItemNames);  

    static ClientError      getMeasures(ClientHandle handle, std::string networkID,
                                bool forwardSearch, int start, int count, 
                                int* startIterNum, int* dataCount, int* curIterNum,
                                int* totalIterNum, float* data);

    static ClientError      getNetworkEvent(ClientHandle handle, std::string networkID,
                                std::vector<NetEvent> &events);

    static ClientError      getNetworkEventMsg(ClientHandle handle, std::string networkID,
                                std::vector<std::string> &msgs);

    static ClientError      checkNetworkDef(ClientHandle handle, std::string networkDef,
                                int &resultCode, int &gpuMBSize, int &layerID,
                                std::string &errorMsg);

    static ClientError      getNetworkProgress(ClientHandle handle, std::string networkID,
                                int &curIterNum, int &totalIterNum);

    static ClientError      getNetworkResult(ClientHandle handle, std::string networkID,
                                std::vector<std::string> &itemNames, 
                                std::vector<float> &itemResults);
};
                            
#endif /* CLIENTAPI_H */
