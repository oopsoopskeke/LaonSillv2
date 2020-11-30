본 문서는 LaonSill python 버전의 사용방법에 대해서 기술한다.

* LaonSill Client Library 사용 방법

(1) LaonSill library를 빌드한다.
$ cd build
$ ./cleanBuildGen.sh
$ ./build_only.sh 12 lib

(2) LaonSill Client Library를 등록한다.
 - 예제) clinet library가 /home/monhoney/laonsill/LaonSill/dev/client/lib에 있는 경우
$ sudo vi /etc/ld.so.conf
/home/monhoney/laonsill/LaonSill/dev/client/lib

(3) $PYTHONPATH 환경변수에 LaonSill pyapi 모듈을 추가한다.
$ export PYTHONPATH=$LAONSILL_SOURCE_PATH/pyapi:$PYTHONPATH


