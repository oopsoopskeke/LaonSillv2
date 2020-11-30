/**
 * @file Examples.cpp
 * @date 2017-06-16
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "Examples.h"
#include "StdOutLog.h"

/**************************************************************************
 * [수정포인트] examples가 정의되어 있는 헤더파일을 include 하도록 합니다
 *************************************************************************/
#include "GAN/GAN.h"
#include "frcnn/FRCNN.h"
#include "LeNet/LeNet.h"
#include "VGG16/VGG16.h"
#include "KISTIKeyword/KISTIKeyword.h"
#include "SSD/SSD.h"
#include "YOLO/YOLO.h"
#include "DenseNet/DenseNet.h"
#include "Inception/Inception.h"
#include "ResNet/ResNet.h"
#include "ZFNet/ZFNet.h"

/**************************************************************************
 * [수정포인트] 추가할 example 개수에 맞게 EXAMPLE_ITEM_DEF_ARRAY_COUNT 값을
 * 변경시켜 줍니다. 
 *************************************************************************/
#define EXAMPLE_ITEM_DEF_ARRAY_COUNT  11 

/**************************************************************************
 * [수정포인트] 추가할 example의 정의를 exampleItemDefArray의 뒷 부분에 기입
 * 합니다. example의 정의는 {"example이름", "example설명", 실행콜백함수}로
 * 이루어져 있습니다. 실행콜백함수는 staic void run() 꼴로 되어 있어야 함니다.
 *************************************************************************/
ExampleItemDef exampleItemDefArray[EXAMPLE_ITEM_DEF_ARRAY_COUNT] = {
    {"GAN", "GAN example", GAN<float>::run},
    {"FRCNN", "FRCNN example", FRCNN<float>::run},
    {"LeNet", "LeNet example", LeNet<float>::run},
    {"VGG16", "VGG16 example", VGG16<float>::run},
    {"KISTIKeyword", "KISTIKeyword example", KISTIKeyword<float>::run},
    {"SSD", "SSD example", SSD<float>::run},
    {"YOLO", "YOLO example", YOLO<float>::run},
    {"DenseNet", "DenseNet example", DenseNet<float>::run},
    {"Inception", "Inception example", Inception<float>::run},
    {"ResNet", "ResNet example", ResNet<float>::run},
    {"ZFNet", "ZFNet example", ZFNet<float>::run}
};


/**************************************************************************
 * 아래 코드 부터는 "[수정포인트]"가 없으니 신경쓰지 않아도 됩니다 :)
 *************************************************************************/
void Examples::checkItem(const char* itemName) {
    for (int i = 0; i < EXAMPLE_ITEM_DEF_ARRAY_COUNT; i++) {
        if (strcmp(itemName, exampleItemDefArray[i].name) == 0)
            return;
    }

    fprintf(stderr, "Invalid item name. item name=%s\n", itemName);
    fprintf(stderr, "available item names are:\n");

    // 많아봤자 몇개나 되겠냐.. 그냥 linear search 하자.
    bool first = true;
    for (int i = 0; i < EXAMPLE_ITEM_DEF_ARRAY_COUNT; i++) {
        if (first) {
            fprintf(stderr, " [%s", exampleItemDefArray[i].name);
            first = false;
        } else {
            fprintf(stderr, ", %s", exampleItemDefArray[i].name);
        }
    }

    fprintf(stderr, "]\n");
    exit(EXIT_FAILURE);
}

void Examples::runItem(ExampleItemDef itemDef) {
    STDOUT_LOG("***************************************************");
    STDOUT_LOG("* %s example", itemDef.name);
    STDOUT_LOG("*  - description : %s", itemDef.desc);

    struct timespec startTime;
    struct timespec endTime;
    clock_gettime(CLOCK_REALTIME, &startTime);
    itemDef.func();
    clock_gettime(CLOCK_REALTIME, &endTime);
    double elapsed = (endTime.tv_sec - startTime.tv_sec) + 
        (endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

    STDOUT_LOG("*  - elapsed time : %lf sec", elapsed);
}

void Examples::run(const char* itemName) {
    // 많아봤자 몇개나 되겠냐.. 그냥 linear search 하자.
    for (int i = 0; i < EXAMPLE_ITEM_DEF_ARRAY_COUNT; i++) {
        if (strcmp(itemName, exampleItemDefArray[i].name) == 0) {
            runItem(exampleItemDefArray[i]);
            break;
        }
    }
    STDOUT_LOG("***************************************************");
}
