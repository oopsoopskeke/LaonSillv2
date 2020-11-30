본 문서는 Log 모듈의 간략한 소개 및 사용방법을 기술한다.

* Log모듈 소개
 4가지 로깅 모듈을 지원하고 있고, 각각의 특성은 아래와 같다:
  - SYS_LOG : 시스템에 중대한 영향을 미치는 중요한 로그를 남긴다.
               ex. critical error, 시스템 시작, 종료
  - COLD_LOG : 일반적인 로깅 시스템. 쉽게 사용할 수 있게 되어 있다.
  - HOT_LOG : 속도에 중점을 둔 로깅 시스템이다. 사용하기 어렵다.
  - STDOUT_BLOCK, STDOUT_LOG : 콘솔 출력을 위한 로깅 시스템이다.

* SYS_LOG
  - SYS_LOG(fmt, arg1, arg2, ...);
  - SYS_LOG가 초기화 되지 않은 시점에는 콘솔에 로그를 출력한다. 
  - fmt, arg1, arg2는 printf의 fmt, arg1, arg2와 동일하다.
  - SPARAM(SYSLOG_DIR)에 지정된 경로에 sys.log 이름으로 로깅파일이 생성된다.
  - SPARAM(SYSLOG_DIR)을 설정하지 않은 경우에는 $LAONSILL_HOME/log 밑에 생성된다.

* COLD_LOG
  - COLD_LOG(level, condition, fmt, arg1, arg2, ...);
  - DEBUG, WARNING, INFO, ERROR 4가지 레벨이 존재 한다.
  - SPARAM(COLDLOG_LEVEL)보다 높게 정의된 로그만 출력한다.
  - condition이 true가 되면 출력 한다.
  - fmt, arg1, arg2는 printf의 fmt, arg1, arg2와 동일하다.
  - SPARAM(COLDLOG_DIR)에 지정된 경로에 sys.log 이름으로 로깅파일이 생성된다.
  - SPARAM(COLDLOG_DIR)을 설정하지 않은 경우에는 $LAONSILL_HOME/log 밑에 생성된다.

* HOT_LOG
  - HOT_LOG(eventId, arg1, arg2, ...); 
  - eventId는 hotCodeDef.json에서 미리 등록이 되어 있어야 한다.
  - arg1, arg2는 hotCodeDef.json에서 지정한 타입들을 순서대로 열거해야 한다.
  - SPARAM(HOTLOG_DIR)에 지정된 경로에 hot.$PID.$TID 이름으로 로깅파일이 생성된다.
    (ex. 17350 processID, 13 TID를 가진 경우 hot.17350.13 이름으로 생성된다.)
  - 바이너리로 기록이 되어 있기 때문에 decodeHotLog.py를 이용해서 디코딩이 필요하다.
  - SPARAM(HOTLOG_BUFFERSIZE)는 메모리 로깅에 사용되는 메모리 크기를 나타낸다.
  - SPARAM(HOTLOG_SLOTSIZE)는 disk I/O를 관리하기 위한 메모리 슬롯의 크기를 나타낸다.
  - SPARAM(HOTLOG_FLUSH_SLOTCOUNT)는 몇개의 메모리 슬롯을 기준으로 flush를 할지를 나타낸다.
  - SPARAM(HOTLOG_FLUSH_CYCLE_MSEC)은 어느 주기(msec)로 플러시를 체크하고 수행하지를 나타낸다.

* STDOUT_BLOCK
  - 사용법 : STDOUT_BLOCK(cout << "hello world : " << abc << endl);

* STDOUT_LOG
  - 사용법 : STDOUT_LOG("Hello World : %d\n", abc);

* HOT_LOG에서 HotCode 클래스 생성방법
  - hotCodeDef.json 파일에 필요한 이벤트ID와 그것의 속성(FMT, ARGS)를 지정한다.
  - 이벤트ID는 정수만 사용할 수 있고, 중복된 정수를 사용하면 안된다.
  - 이벤트ID에서 0값은 reserved되어 있기 때문에 바꾸면 안된다.
  - genHotCode.py를 실행하여 HotCode.h, HotCode.cpp 파일을 생성한다.

* 디코딩 방법
 -  "(1) 특정 PID에 대한 모든 파일을 디코딩하는 방법"과 "(2) 특정 PID, TID에 대한 파일을 디코딩하는
 방법" 2가지가 있다.
 - (1)의 경우에는 "decodeHotLog.py hotCodeDefFilePath hotLogTopDir pid"를 수행한다.
 - (2)의 경우에는 "decodeHotLog.py hotCodeDefFilePath hotLogTopDir pid tid"를 수행한다.
 - hotCodeDefFilePath는 HotCode 클래스 생성시에 사용이 되었던 hotCodeDef.json 파일의 경로를
 가리킨다.
 - hotLogTopDir은 저장된 로그가 존재하고 있는 디렉토리를 말한다.
 - pid는 HOT LOG를 남길때 해당 프로그램의 process ID를 가리킨다.
 - tid는 해당 프로그램의 쓰레드 ID를 가리킨다.

