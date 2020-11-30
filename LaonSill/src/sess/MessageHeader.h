/**
 * @file MessageHeader.h
 * @date 2016-10-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef MESSAGEHEADER_H
#define MESSAGEHEADER_H 

#include "common.h"

// Serialize시 packet 구성
// |---------+--------+------------------------------------------|
// | MsgType | MsgLen | MsgBody                                  |
// | int(4)  | int(4) | variable size => MsgLen - 8 byte         |
// |---------+--------+------------------------------------------|
// |<-- msg header -->|

class MessageHeader {
public:
    static const int                    MESSAGE_HEADER_SIZE = 8;
    static const int                    MESSAGE_DEFAULT_SIZE = 1452;
    // usual MTU size(1500) - IP header(20) - TCP header(20) - vpn header(8)
    // 물론 MTU 크기를 변경하면 더 큰값을 설정해도 된다..
    // 하지만.. 다른 네트워크의 MTU 크기가 작다면 성능상 문제가 발생할 수 밖에 없다.

    enum MsgType : int {
        Welcome = 0,
        WelcomeReply,

        PushJob = 10,
        PushJobReply,

        GoodBye = 90,

        HaltMachine = 100,
    };

                MessageHeader() {}
    virtual    ~MessageHeader() {}

    int         getMsgLen() { return this->msgLen; }
    MsgType     getMsgType() { return this->msgType; }

    void        setMsgType(MsgType msgType) { this->msgType = msgType; }
    void        setMsgLen(int msgLen) { this->msgLen = msgLen; }

private:
    MsgType     msgType;
    int         msgLen;
};

#endif /* MESSAGEHEADER_H */
