/**
 * @file ThreadMgmt.h
 * @date 2017-06-08
 * @author moonhoen lee
 * @brief 쓰레드를 관리하는 모듈
 * @details
 */

#ifndef THREADMGMT_H
#define THREADMGMT_H 

#include <string>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <atomic>

typedef enum ThreadType_s {
    Producer,
    TaskConsumer,
    JobConsumer,
    Sender,
    Receiver,
    ThreadTypeMax
} ThreadType;

// bitwise operation을 할 수 있기 때문에 2^n 꼴로 정의 해야 함.
typedef enum ThreadEvent_e : unsigned long {
    Wakeup = 1UL,
    Timeout = 2UL,
    Halt = 4UL,
    FinishJob = 8UL
} ThreadEvent;

typedef struct ThreadContext_s {
    int threadID;
    std::string name;   /* for debug */
    std::mutex mutex;
    std::condition_variable cv;
    volatile bool flag;     // 깨울때에 true로 설정이 되고, 쓰레드가 깨어나면 false로 바꾼다.
                            // 쓰래드가 수행해야 할 일을 한 루프돌고 나서 true이면 다시한번
                            // 도는 방식이다.
                            // 이 플래그는 논리 흐름상 락으로 보호하지 않아도 된다.
                            // 왜냐면 돌지 않아도 될 타이밍에 한번 더 돌아도 문제가 없고,
                            // 돌아야할 타이밍에 돌지 않은 경우는 없기 때문이다.

    std::atomic<unsigned long> event;   // 어떠한 이벤트로 쓰레드가 깨워지게 되었는지를
                                        // 정의하는 항목
} ThreadContext;

class ThreadMgmt {
public: 
    ThreadMgmt() {}
    virtual ~ThreadMgmt() {}

    static int init();
    static void destroy();
    static void signal(int threadID, ThreadEvent event);
    static void signalAll(ThreadEvent event);
    static ThreadEvent wait(int threadID, unsigned long timeout);
    static int getThreadID(ThreadType type, int offset);
    static void setThreadReady(int threadID);
    static bool isReady();

private:
    static int                          threadCount;
    static std::vector<ThreadContext*>  contextArray;
    static int                         *threadIDBaseArray;
    static volatile std::atomic<int>    readyCount;
};
#endif /* THREADMGMT_H */
