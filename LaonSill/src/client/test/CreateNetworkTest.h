/**
 * @file CreateNetworkTest.h
 * @date 2017-06-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef CREATENETWORKTEST_H
#define CREATENETWORKTEST_H 
class CreateNetworkTest {
public: 
    CreateNetworkTest() {}
    virtual ~CreateNetworkTest() {}
    static bool runTest();

private:
    static bool runSimpleTest();
    static bool runCreateNetworkFromFileTest();
};
#endif /* CREATENETWORKTEST_H */
