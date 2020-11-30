/**
 * @file MsgSerializer.h
 * @date 2016-10-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef SERIALIZER_H
#define SERIALIZER_H 

#include "common.h"
#include "MessageHeader.h"

class MsgSerializer {
public:
                MsgSerializer() {}
    virtual    ~MsgSerializer() {}
   
    static int  serializeInt(int data, int offset, char* msg);
    static int  deserializeInt(int& data, int offset, char* msg);

    static int  serializeFloat(float data, int offset, char* msg);
    static int  deserializeFloat(float& data, int offset, char* msg);

    static int  serializeString(const char* data, int len, int offset, char* msg);
    static int  deserializeString(char* data, int len, int offset, char* msg);

    static int  serializeMsgHdr(MessageHeader msgHdr, char* msg);
    static int  deserializeMsgHdr(MessageHeader& msgHdr, char* msg);

};

#endif /* SERIALIZER_H */
