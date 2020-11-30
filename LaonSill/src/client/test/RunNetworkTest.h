/**
 * @file RunNetworkTest.h
 * @date 2017-06-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef RUNNETWORKTEST_H
#define RUNNETWORKTEST_H 

class RunNetworkTest {
public: 
    RunNetworkTest() {}
    virtual ~RunNetworkTest() {}
    static bool runTest();

private:
    static bool runSimpleTest();
    static bool runMiniBatchTest();
    static bool runTwiceTest();
};

#endif /* RUNNETWORKTEST_H */
