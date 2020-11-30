/**
 * @file MsgSerializer.cpp
 * @date 2016-10-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string.h>
#include <arpa/inet.h>

#include "MsgSerializer.h"

using namespace std;

int MsgSerializer::serializeInt(int data, int offset, char* msg) {
    int temp;
    temp = htonl(data);
    memcpy((void*)(msg + offset), (void*)&temp, sizeof(int));
   
    return offset + sizeof(int);
}

int MsgSerializer::deserializeInt(int& data, int offset, char* msg) {
    int temp;
    memcpy((void*)&temp, (void*)(msg + offset), sizeof(int));
    data = ntohl(temp);
   
    return offset + sizeof(int);
}

int MsgSerializer::serializeFloat(float data, int offset, char* msg) {
    // XXX: float는 architecture마다 다를 수 있다.
    //      일단 표준인 IEEE 754-1985를 따른다고 가정하고 코딩하였다.
    //      float를 int로 캐스팅하면 float value가 truncate되기 때문에
    //      포인터로 넘겨주는 방식으로 구현하였다.
    int temp;
    temp = htonl(*(int*)&data);
    memcpy((void*)(msg + offset), (void*)&temp, sizeof(int));
   
    return offset + sizeof(int);
}

int MsgSerializer::deserializeFloat(float& data, int offset, char* msg) {
    int     temp;
    float   tempFloatData;
    memcpy((void*)&temp, (void*)(msg + offset), sizeof(int));
    *(int*)&tempFloatData = ntohl(temp);
    data = tempFloatData;
   
    return offset + sizeof(int);
}

int MsgSerializer::serializeString(const char* data, int len, int offset, char* msg) {
    memcpy((void*)(msg + offset), (void*)data, sizeof(char) * len);
    return offset + sizeof(char) * len;
}

int MsgSerializer::deserializeString(char* data, int len, int offset, char* msg) {
    memcpy((void*)data, (void*)(msg + offset), sizeof(char) * len);
    return offset + sizeof(char) * len;
}

int MsgSerializer::serializeMsgHdr(MessageHeader msgHdr, char* msg) {
    // @See: MessageHeader.h
    int offset = 0;
    offset = MsgSerializer::serializeInt(msgHdr.getMsgType(), offset, msg);
    offset = MsgSerializer::serializeInt(msgHdr.getMsgLen(), offset, msg);
    return offset;
}

int MsgSerializer::deserializeMsgHdr(MessageHeader& msgHdr, char* msg) {
    // @See: MessageHeader.h
    int msgType;
    int msgLen;
    int offset = 0;

    offset = MsgSerializer::deserializeInt(msgType, offset, msg); 
    offset = MsgSerializer::deserializeInt(msgLen, offset, msg); 

    msgHdr.setMsgType((MessageHeader::MsgType)msgType);
    msgHdr.setMsgLen(msgLen);

    return offset;
}
