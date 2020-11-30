/**
 * @file RunNetworkWithInputTest.h
 * @date 2017-07-13
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef RUNNETWORKWITHINPUTTEST_H
#define RUNNETWORKWITHINPUTTEST_H 

class RunNetworkWithInputTest {
public: 
    RunNetworkWithInputTest() {}
    virtual ~RunNetworkWithInputTest() {}
    static bool runTest();

private:
    static bool runSimpleTest();
};

#endif /* RUNNETWORKWITHINPUTTEST_H */
