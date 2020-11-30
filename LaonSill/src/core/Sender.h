/**
 * @file Sender.h
 * @date 2017-06-08
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef SENDER_H
#define SENDER_H 

#include <thread>

class Sender {
public: 
    Sender() {}
    virtual ~Sender() {}
    static void             senderThread();
	static void             launchThread();
private:
    static std::thread*     sender;
};
#endif /* SENDER_H */
