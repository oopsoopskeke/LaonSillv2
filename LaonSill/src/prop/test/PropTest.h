/**
 * @file PropTest.h
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef PROPTEST_H
#define PROPTEST_H 
class PropTest {
public: 
    PropTest() {}
    virtual ~PropTest() {}

    static bool runTest();
private:
    static bool runSimpleLayerPropTest();
    static bool runSimpleNetworkPropTest();
};
#endif /* PROPTEST_H */
