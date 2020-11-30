본 문서는 Perf 모듈의 간략한 소개 및 사용방법을 기술한다.

* Perf 모듈이란?
 Performance를 측정하기 위한 모듈이다. 
 특정 구간에 대한 call count, time, average time, max time을 측정할 수 있다.
 추가적으로 해당 구간에 의미있는 인자들을 정의하여 그 값을 축적시킬 수 있다. 예를 들면,
I/O시간을 측정한다고 하였을 시에 I/O 크기값을 추가 인자로 정의하여 그 값을 축적시킬 수 있다.

* Performance 등록 방법
 perfDef.json 파일에 등록할 Performance 이름과 그것의 6가지 속성을 json format에 맞게
 기입한다.

 - 2개 performance(SERVER_RUNNING_TIME, NETWORK_TRAINING_TIME) 추가 예제
 [perfDef.json]
     :
"SERVER_RUNNING_TIME" :
{
    "DESC"          : "specify server running time",
    "SCOPE"         : "SYSTEM",
    "USETIME"       : true,
    "USEAVGTIME"    : false,
    "USEMAXTIME"    : false,
    "ARGS"          : []
},

"NETWORK_TRAINING_TIME" :
{
    "DESC"          : "specify network training time",
    "SCOPE"         : "JOB",
    "USETIME"       : true,
    "USEAVGTIME"    : false,
    "USEMAXTIME"    : false,
    "ARGS"          :
    [
        ["EpochCount, "UINT32", "specify epoch count"],
        ["GPUCount, "UINT32", "specify GPU count"]
    ]
}

}

* 각각의 속성에 대한 설명
총 6개의 속성이 정의 된다. 설명은 아래와 같다:
(1) DESC : 퍼포먼스에 대한 설명이다. 영어로 작성한다.
(2) SCOPE : 퍼포먼스가 해석이 되는 범위를 정의
            "JOB"인 경우에는 Worker Thread별로 퍼포먼스가 측정이 된다.
            "SYSTEM"인 경우에는 시스템 전체적으로 wall clock으로 측정 된다.
            참고로 "JOB"인 경우에 자신이 원하는 CLOCK_TYPE을 고를 수 있다.
            (※ SPARAM(JOBSCOPE_CLOCKTYPE) 참고 바람)
(3) USETIME : 구간에 걸린 시간에 대한 측정 여부를 결정한다.
              true인 경우에는 아래와 같이 측정 한다:
            {
              struct timespec startTime; 
              SPERF_START(perfName, &startTime);  
              // 측정 구간
                   ....
              SPERF_END(perfName, startTime, args...);
              cout << "elapsed time : " << SPERF_TIME(perfName) << endl;
            }
              false인 경우에는 아래와 같이 측정 한다:
            { 
              for (int i = 0 ; i < 100; i++) {
                  SPERF_MARK(perfName);
                   :
                   :
              }
              cout << "call count : " << SPERF_MARK(perfName) << endl;
            }
(4) USEAVGTIME : 구간에 걸린 평균 시간에 대한 측정 여부를 결정한다.
                 반드시 USETIME=true이어야 동작한다.
(5) USEMAXTIME : 구간에 걸린 최대 시간에 대한 측정 여부를 결정한다.
                 반드시 USETIME=true이어야 동작한다.
(6) ARGS : 가변인자에 대한 정보를 정의 한다.
           가변인자 타입은 정수형(ex. INT8, INT32, UINT64)와 실수형(ex. double, float)에
          대해서만 지원한다.
           (가변인자명, 가변인자 타입, 가변인자 설명)의 형태로 적는다.

* 퍼포먼스 리스트가 정의된 파일(PerfList.h, PerfList.cpp) 생성방법
genPerf.py를 실행한다.
