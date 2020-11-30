/**
 * @file BrokerTest.h
 * @date 2016-12-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef BROKERTEST_H
#define BROKERTEST_H 

class BrokerTest {
public: 
    BrokerTest() {}
    virtual ~BrokerTest() {}

    static bool runTest();
private:
    static bool runSimplePubSubTest();
    static bool runBlockingPubSubTest();
};

#endif /* BROKERTEST_H */
