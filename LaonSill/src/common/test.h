/**
 * @file test.h
 * @date 2016-12-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef TEST_H
#define TEST_H

#define TEST_ITEM_DEF_NAME_MAXLEN   (256)
#define TEST_ITEM_DEF_DESC_MAXLEN   (256)

typedef bool(*CBTestFunc)();

typedef struct TestItemDef_s {
   char         name[TEST_ITEM_DEF_NAME_MAXLEN];
   char         desc[TEST_ITEM_DEF_DESC_MAXLEN];
   CBTestFunc   func; 
} TestItemDef;

void checkTestItem(const char* testItemName);
void runTestItem(TestItemDef itemDef);
void runTest(const char* testItemName);

#endif /* TEST_H */
