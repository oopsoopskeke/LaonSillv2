본 문서는 Param Class에 대한 간략한 설명을 담고 있다.

* 최초 실행시 빌드 방법
(1) genParam.py를 실행하여 Param.cpp, Param.h 파일을 생성한다.
$ ./genParam.py

(2) makefile을 수정하여 빌드를 한다.
    makefile에 Param class에 대한 정보가 없기 때문에 빌드에 문제가 된다.
    그래서 makefile을 Param class를 고려하여 빌드하도록 수정해야 한다.
    여러가지 방법이 있겠지만 nsight에서 makefile을 자동 생성하는 방식을 추천한다.
    nsight에서 파일 refresh 이후에 빌드를 하면 Param class가 고려된 makefile이 만들어 진다.

* 파라미터 추가 방법
paramDef.json 파일에 파라미터 이름과 그것의 6가지 속성을 json format에 맞게 기입한다. 

 - 2개 parameter(BEST_SOCCER_PLAYER_NAME, SOCCER_PLAYER_RETIRE_AGE) 추가 예제
[paramDef.json]
            :
"BEST_SOCCER_PLAYER_NAME" : 
{
        "DESC" : "Specify best soccer player's name",
        "MANDATORY" : false,
        "MUTABLE" : false,
        "SCOPE" : "SYSTEM",
        "TYPE" : "CHAR(128)",
        "DEFAULT" : "moonhoen lee" 
},

"SOCCER_PLAYER_RETIRE_AGE" : 
{
        "DESC" : "Specify soccer players' retire age",
        "MANDATORY" : false,
        "MUTABLE" : true,
        "SCOPE" : "SESSION",
        "TYPE" : "UINT32",
        "DEFAULT" : 42 
}

}

* 파라미터 설명
총 6개의 파라미터가 있다. 설명은 아래와 같다:
(1) DESC : 파라미터에 대한 설명이며, 영어로 작성 한다.
           (문법적인 측면은 고려하지 말고, 이걸로 스트레스를 절대 받지 말자.
            구글번역기 돌려서 기입해도 무방하다.
            어차피 제품 출시전에 전문가에게 의뢰해서 다 수정해야 한다.)
(2) MANDATORY : 초기파라미터 설정파일에 파라미터 값을 반드시 기입을 해야 하는지 여부를 정의
(3) MUTABLE : 파라미터 값을 runtime에 변경 할 수 있는지 여부를 정의
(4) SCOPE : 파라미터가 해석이 되는 범위를 정의. 
            "SYSTEM"은 모든 세션에 파라미터 값이 적용이 되고, "SESSION"은 개별 세션별로
           다른 파라미터 값을 적용할 수 있다는 것을 의미.
(5) TYPE : 파라미터 값의 타입을 정의.
           정수형으로는 INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64를 지원
           실수형으로는 FLOAT, DOUBLE, LONGDOUBLE을 지원
           문자열형은 반드시 문자열 최대 길이를 명시해야 함. (ex. CHAR(32))
           그 외에 BOOL 형을 지원 (true/false)
(6) DEFAULT : 기본 값을 정의. MANDATORY가 false이면 반드시 기본 값을 명시해야 함.
              MANDATORY가 true이면 DEFAULT는 ""으로 적어야 함.

※ 주의사항
(1) Param.cpp, Param.h 파일은 genParam.py에 의하여 자동 생성이 되는 파일들이다. 손으로 직접
  해당 파일들을 수정하면 안된다. 반드시 paramDef.json파일을 수정하고, genParam.py를 실행하는
  방식으로 수정해야 한다.

(2) Param.cpp, Param.h파일은 git에 관리되지 않아야 한다. genParam.py와 paramDef.json 파일만
  관리 되도록 주의한다.

(3) 빌드시에 make clean으로 확실히 지워주고 빌드하지 않으면 오동작하는 경우를 발견하였다. 
  (원인은 아직 파악하지 못함.) 반드시 make clean하고 사용하자.
