/**
 * @file SaveLoadNetworkTest.h
 * @date 2017-06-23
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef SAVELOADNETWORKTEST_H
#define SAVELOADNETWORKTEST_H 
class SaveLoadNetworkTest {
public: 
    SaveLoadNetworkTest() {}
    virtual ~SaveLoadNetworkTest() {}
    static bool runTest();

private:
    static bool runSaveTest();
    static bool runLoadTest();
};
#endif /* SAVELOADNETWORKTEST_H */
