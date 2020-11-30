#!/usr/bin/bash

cd $LAONSILL_BUILD_PATH

echo "[generate parameter]"
cd src/param
./genParam.py
if [ "$?" -ne 0 ]; then
    echo "ERROR: build stopped"
    exit -1
fi
cd ../..

echo "[generate enum def]"
cd src/prop
./genEnum.py
if [ "$?" -ne 0 ]; then
    echo "ERROR: build stopped"
    exit -1
fi
cd ../..

echo "[generate layer prop]"
cd src/prop
./genLayerPropList.py
if [ "$?" -ne 0 ]; then
    echo "ERROR: build stopped"
    exit -1
fi
cd ../..

echo "[generate network prop]"
cd src/prop
./genNetworkProp.py
if [ "$?" -ne 0 ]; then
    echo "ERROR: build stopped"
    exit -1
fi
cd ../..

echo "[generate hotcode]"
cd src/log
./genHotCode.py
if [ "$?" -ne 0 ]; then
    echo "ERROR: build stopped"
    exit -1
fi
echo "[copy tools related hotcode]"
cp decodeHotLog.py ../../bin/.
cp hotCodeDef.json ../../bin/.
cd ../..

echo "[generate performance]"
cd src/perf
./genPerf.py
if [ "$?" -ne 0 ]; then
    echo "ERROR: build stopped"
    exit -1
fi
cd ../..
