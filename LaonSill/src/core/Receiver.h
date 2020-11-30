/**
 * @file Receiver.h
 * @date 2017-06-08
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef RECEIVER_H
#define RECEIVER_H 

#include <thread>

class Receiver {
public: 
    Receiver() {}
    virtual ~Receiver() {}
    static void         receiverThread();
	static void         launchThread();
private:
    static std::thread* receiver;
};
#endif /* RECEIVER_H */
