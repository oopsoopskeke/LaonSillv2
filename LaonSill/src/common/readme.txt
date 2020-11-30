* 테스트모드 실행 방법
(1) 모든 테스트 실행하기
$ ./LaonSillServer -t all

(2) 개별 테스트 실행하기
 [ex. broker test 실행하기]
$ ./LaonSillServer -t broker

* 테스트를 추가 방법
(1) 테스트 클래스 생성
 - 모듈 폴더(ex. core, client, network) 밑에 test라는 폴더를 만들고, 그곳에 AAATest.h & AAATest.cpp 형태로 생성합니다.

(2) 테스트 함수 작성
 - bool runTest(); 형태로 정의합니다.
 - return값이 true이면 성공, false이면 실패 입니다.
 - 굳이 테스트 함수의 인자를 정의할 필요성이 없어서 비워 두었습니다.
   (혹시 필요하면 알려주십시요)

(3) common/test.cpp에 테스트 등록
 - 아래의 3개의 "수정포인트"에 주석을 참고하여 추가/수정 합니다.
 - 나중에 파이썬으로 빌드시에 자동으로 등록할 수 있도록 스크립트를 만들 계획은 있습니다.... (좀... 한참후에.;;;)
