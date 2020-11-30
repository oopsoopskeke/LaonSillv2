/**
 * @file Update.h
 * @date 2017-05-30
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef UPDATE_H
#define UPDATE_H 

#include "common.h"
#include "Data.h"
#include "EnumDef.h"

typedef struct UpdateContext_s {
    int     paramSize;
    float   regScale;
    float   learnScale;
    float   epsilon;
    float   decayRate;
    float   beta1;
    float   beta2;
    float   decayedBeta1;
    float   decayedBeta2;
} UpdateContext;

typedef struct UpdateParam_s {
    int             paramType;
    UpdateContext   context;
    void*           paramDataPtr;
    void*           paramHis1Ptr;
    void*           paramHis2Ptr;
} UpdateParam;

template<typename Dtype>
class Update {
public: 
    Update() {}
    virtual ~Update() {}

    static void updateParam(UpdateContext context, Data<Dtype>* dataHistory,
        Data<Dtype>* dataHistory2, Data<Dtype>* data);

    static void doNesterov(int size, const Dtype* dx, Dtype* v_prev, Dtype* v, Dtype* x,
        const Dtype mu, const Dtype lr);

    static void doAdagrad(int size, const Dtype* dx, Dtype* cache, Dtype* x,
        const Dtype lr, const Dtype eps);

    static void doRMSprop(int size, const Dtype* dx, Dtype* cache, Dtype* x,
        const Dtype lr, const Dtype eps, const Dtype dr);

    static void doAdam(int size, const Dtype* dx, Dtype* m, Dtype* v, Dtype* x,
        const Dtype lr, const Dtype eps, const Dtype beta1, const Dtype beta2,
        const Dtype decayedBeta1, const Dtype decayedBeta2);

    static void doAdadelta(int size, const Dtype* dx, Dtype* e1, Dtype* e2, Dtype* x,
            const Dtype mu, const Dtype lr, const Dtype eps);

    static float calcLearningRate();

    static UpdateContext makeContext(int paramSize, float regScale, float learnScale);

    static int getParamHistoryDataCount(Optimizer opt);

    static void clipGradients(const int size, Dtype* dx);
};

#endif /* UPDATE_H */
