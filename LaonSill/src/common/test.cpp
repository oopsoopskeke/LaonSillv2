/**
 * @file test.cpp
 * @date 2016-12-21
 * @author moonhoen lee
 * @brief 
 * @details
 *  나중에 시간나면 python으로 자동 generate하도록 변경하겠음 :)
 *  지금은 좀 귀찮아서 ㅠ_ㅜ;;;
 *
 *  테스트를 추가하고 싶은 경우에 [수정포인트] 라고 쓰여진 부분의 주석을 참고하여
 *  추가하면 됩니다
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>

#include "test.h"
#include "StdOutLog.h"

#define gettid()        syscall(SYS_gettid)

/**************************************************************************
 * [수정포인트] test가 정의되어 있는 헤더파일을 include 하도록 합니다
 * 각 test 클래스들은 모듈 폴더(ex. core, network, ..)밑에 위치하게 됩니다.
 * 모듈 폴더와 달리 test 폴더들은 include 패스를 추가하지 않았습니다.
 * 따라서, test라는 폴더명을 붙이고, include path를 추가해야 한다는 점을
 * 유의 바랍니다.
 *************************************************************************/
#ifndef CLIENT_MODE
#include "test/BrokerTest.h"
#include "test/PropTest.h"
#include "test/PlanParserTest.h"
#include "test/PlanOptimizerTest.h"
#include "test/PlanBuilderTest.h"
#include "test/NetworkRunByPlanTest.h"
#include "test/CustomInputTest.h"
#else
#include "test/CreateNetworkTest.h"
#include "test/RunNetworkTest.h"
#include "test/SaveLoadNetworkTest.h"
#include "test/RunNetworkWithInputTest.h"
#include "test/PrintMeasureTest.h"
#endif

/**************************************************************************
 * [수정포인트] 추가할 테스트 개수에 맞게 TEST_ITEM_DEF_ARRAY_COUNT 값을
 * 변경시켜 줍니다. 테스트 개수에 all이 포함이 되지는 않습니다.
 *************************************************************************/
#ifndef CLIENT_MODE
#define TEST_ITEM_DEF_ARRAY_COUNT  7
#else
#define TEST_ITEM_DEF_ARRAY_COUNT  5
#endif

/**************************************************************************
 * [수정포인트] 추가할 테스트의 정의를 testItemDefArray의 뒷 부분에 기입
 * 합니다. 테스트의 정의는 {"테스트이름", "테스트설명", 테스트콜백함수}로
 * 이루어져 있습니다.
 *************************************************************************/
#ifndef CLIENT_MODE
TestItemDef testItemDefArray[TEST_ITEM_DEF_ARRAY_COUNT] = {
    {"broker", "subscribe & publish function test", BrokerTest::runTest},
    {"prop", "layer prop function test", PropTest::runTest},
    {"planparser", "plan parser function test", PlanParserTest::runTest},
    {"planopt", "plan optimizer function test", PlanOptimizerTest::runTest},
    {"planbuilder", "plan builder function test", PlanBuilderTest::runTest},
    {"runbyplan", "running network by plantype function test", NetworkRunByPlanTest::runTest},
    {"custominput", "custom input function test", CustomInputTest::runTest}
};
#else
TestItemDef testItemDefArray[TEST_ITEM_DEF_ARRAY_COUNT] = {
    {"create", "create network test", CreateNetworkTest::runTest},
    {"run", "run network test", RunNetworkTest::runTest},
    {"saveload", "save & load network test", SaveLoadNetworkTest::runTest},
    {"telco", "run network with input data test", RunNetworkWithInputTest::runTest},
    {"measure", "print measure module test", PrintMeasureTest::runTest}
};
#endif

/**************************************************************************
 * 아래 코드 부터는 "[수정포인트]"가 없으니 신경쓰지 않아도 됩니다 :)
 *************************************************************************/
void checkTestItem(const char* testItemName) {
    if (strcmp(testItemName, "all") == 0)
        return;

    for (int i = 0; i < TEST_ITEM_DEF_ARRAY_COUNT; i++) {
        if (strcmp(testItemName, testItemDefArray[i].name) == 0)
            return;
    }

    fprintf(stderr, "Invalid test item name. item name=%s\n", testItemName);
    fprintf(stderr, "available test item names are:\n");

    // 테스트 많아봤자 몇개나 되겠냐.. 그냥 linear search 하자.
    bool first = true;
    for (int i = 0; i < TEST_ITEM_DEF_ARRAY_COUNT; i++) {
        if (first) {
            fprintf(stderr, " [%s", testItemDefArray[i].name);
            first = false;
        } else {
            fprintf(stderr, ", %s", testItemDefArray[i].name);
        }
    }

    fprintf(stderr, ", all]\n");
    exit(EXIT_FAILURE);
}

void runTestItem(TestItemDef itemDef) {
    STDOUT_LOG("***************************************************");
    STDOUT_LOG("* %s test", itemDef.name);
    STDOUT_LOG("*  - description : %s", itemDef.desc);

    struct timespec startTime;
    struct timespec endTime;
    clock_gettime(CLOCK_REALTIME, &startTime);
    bool testResult = itemDef.func();
    clock_gettime(CLOCK_REALTIME, &endTime);
    double elapsed = (endTime.tv_sec - startTime.tv_sec) + 
        (endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

    STDOUT_LOG("*  - elapsed time : %lf sec", elapsed);
    if (testResult) {
        STDOUT_LOG("*  - result : success");
    } else {
        STDOUT_LOG("*  - result : fail");
        exit(EXIT_FAILURE);
    }
}

// 이 함수에 들어오기전에 반드시 checkTestItem()이 호출 되어 있어야 한다.
void runTest(const char* testItemName) {
    if (strcmp(testItemName, "all") == 0) {
        for (int i = 0; i < TEST_ITEM_DEF_ARRAY_COUNT; i++) {
            runTestItem(testItemDefArray[i]);
        }
    } else {
        // 테스트 많아봤자 몇개나 되겠냐.. 그냥 linear search 하자.
        for (int i = 0; i < TEST_ITEM_DEF_ARRAY_COUNT; i++) {
            if (strcmp(testItemName, testItemDefArray[i].name) == 0) {
                runTestItem(testItemDefArray[i]);
                break;
            }
        }
    }
    STDOUT_LOG("***************************************************");
}
