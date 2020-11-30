/**
 * @file NetworkRunByPlanTest.h
 * @date 2017-06-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef NETWORKRUNBYPLANTEST_H
#define NETWORKRUNBYPLANTEST_H 
class NetworkRunByPlanTest {
public: 
    NetworkRunByPlanTest() {}
    virtual ~NetworkRunByPlanTest() {}
    static bool runTest();

private:
    static bool runPlanOnceTest();

};
#endif /* NETWORKRUNBYPLANTEST_H */
