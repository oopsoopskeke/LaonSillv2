#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from LaonSill.ClientAPI import *
import os
import time

lenet_train_networkdef = \
"""
{
	"layers" :
	[
		{
			"name" : "data",
			"layer" : "DataInput",
			"id" : 0,
			"output" : ["data", "label"],
			"source" : "$(LAONSILL_HOME)/data/sdf/mnist_train_sdf",
			"dataSetName" : "train",
			"dataTransformParam.scale" : 0.00390625
		},

		{
			"name" : "conv1",
			"layer" : "Conv",
			"id" : 10,
			"input" : ["data"],
			"output" : ["conv1"],
			"filterDim.rows" : 5,
			"filterDim.cols" : 5,
			"filterDim.channels" : 1,
			"filterDim.filters" : 20,
			"filterDim.pad" : 0,
			"filterDim.stride" : 1,
			"weightUpdateParam.lr_mult" : 1.0,
			"weightUpdateParam.decay_mult" : 1.0,
			"biasUpdateParam.lr_mult" : 2.0,
			"biasUpdateParam.decay_mult" : 1.0,
			"weightFiller.type" : "Xavier",
			"biasFiller.type" : "Constant"
		},

		{
			"name" : "pool1",
			"layer" : "Pooling",
			"id" : 20,
			"input" : ["conv1"],
			"output" : ["pool1"],
			"poolDim.rows" : 2,
			"poolDim.cols" : 2,
			"poolDim.pad" : 0,
			"poolDim.stride" : 2,
			"poolingType" : "Max"
		},

		{
			"name" : "conv2",
			"layer" : "Conv",
			"id" : 30,
			"input" : ["pool1"],
			"output" : ["conv2"],
			"filterDim.rows" : 5,
			"filterDim.cols" : 5,
			"filterDim.channels" : 20,
			"filterDim.filters" : 50,
			"filterDim.pad" : 0,
			"filterDim.stride" : 1,
			"weightUpdateParam.lr_mult" : 1.0,
			"weightUpdateParam.decay_mult" : 1.0,
			"biasUpdateParam.lr_mult" : 2.0,
			"biasUpdateParam.decay_mult" : 1.0,
			"weightFiller.type" : "Xavier",
			"biasFiller.type" : "Constant"
		},

		{
			"name" : "pool2",
			"layer" : "Pooling",
			"id" : 40,
			"input" : ["conv2"],
			"output" : ["pool2"],
			"poolDim.rows" : 2,
			"poolDim.cols" : 2,
			"poolDim.pad" : 0,
			"poolDim.stride" : 2,
			"poolingType" : "Max"
		},
		
		{
			"name" : "ip1",
			"layer" : "FullyConnected",
			"id" : 50,
			"input" : ["pool2"],
			"output" : ["ip1"],
			"nOut" : 500,
			"weightUpdateParam.lr_mult" : 1.0,
			"weightUpdateParam.decay_mult" : 1.0,
			"biasUpdateParam.lr_mult" : 2.0,
			"biasUpdateParam.decay_mult" : 1.0,
			"weightFiller.type" : "Xavier",
			"biasFiller.type" : "Constant"			
		},

		{
			"name" : "relu1",
			"layer" : "Relu",
			"id" : 60,
			"input" : ["ip1"],
			"output" : ["ip1"]
		},

		{
			"name" : "ip2",
			"layer" : "FullyConnected",
			"id" : 70,
			"input" : ["ip1"],
			"output" : ["ip2"],
			"nOut" : 10,
			"weightUpdateParam.lr_mult" : 1.0,
			"weightUpdateParam.decay_mult" : 1.0,
			"biasUpdateParam.lr_mult" : 2.0,
			"biasUpdateParam.decay_mult" : 1.0,
			"weightFiller.type" : "Xavier",
			"biasFiller.type" : "Constant"			
		},
		
		{
			"name" : "loss",
			"layer" : "SoftmaxWithLoss",
			"id" : 80,
			"input" : ["ip2", "label"],
			"output" : ["loss"],
			"propDown" : [true, false],
			"softmaxAxis" : 2
		},
		
		{
			"name" : "accuracy",
			"layer" : "Accuracy",
			"id" : 90,
			"input" : ["ip2", "label"],
			"output" : ["accuracy"],
			"propDown" : [false, false],
			"topK" : 1,
			"axis" : 2
		}
	],

	"configs" :
	{
		"batchSize" : 64,
		"epochs" : 10,
		"lossLayer" : ["loss"],
		"testInterval" : 1000,
		"saveInterval" : 1000,
		"savePathPrefix" : "", 
		"baseLearningRate" : 0.01,
		"weightDecay" : 0.0005,
		"momentum" : 0.9,
		"lrPolicy" : "Inv",
		"gamma" : 0.0001,
		"power" : 0.75,
		"optimizer" : "Momentum",
        "status" : "Train",
        "measureLayer" : ["loss", "accuracy"],
        "keepSaveIntervalModelCount" : 4,
        "keepSaveBestModel" : false,
        "keepSaveBestModelStartIterNum" : 10000
	}
}
"""

# (1) LaonSill Client 핸들을 생성한다.
print "create handle"
handle = ClientHandle()
ret = handle.createHandle()

# (2) 세션을 얻는다.
print "get session"
ret = handle.getSession()

# (3) 네트워크 정의 파일을 체크한다.
print "check network def"
ret, result_code, gpu_MB_size, layer_id, msg = handle.checkNetworkDef(lenet_train_networkdef)
print "ret : ", ret, ", resultCode : ", result_code, ", gpuMBSize : ", gpu_MB_size,\
    ", layerID : ", layer_id, ", msg : ", msg

# (4) 세션 자원을 정리합니다.
print "cleanup resources"
handle.releaseSession()
