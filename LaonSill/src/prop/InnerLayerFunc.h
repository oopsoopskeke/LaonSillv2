/**
 * @file InnerLayerFunc.h
 * @date 2017-05-26
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef INNERLAYERFUNC_H
#define INNERLAYERFUNC_H 
class InnerLayerFunc {
public: 
    static void initLayer(int innerLayerIdx);
    static void destroyLayer(int innerLayerIdx);
    static void setInOutTensor(int innerLayerIdx, void *tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(int innerLayerIdx);
    static void runForward(int innerLayerIdx, int miniBatchIdx);
    static void runBackward(int innerLayerIdx);
    static void learn(int innerLayerIdx);

    InnerLayerFunc() {}
    virtual ~InnerLayerFunc() {}
};
#endif /* INNERLAYERFUNC_H */
