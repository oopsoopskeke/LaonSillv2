본 문서는 Prop 모듈의 간략한 소개 및 사용방법을 기술한다.

* Prop 모듈이란?
  enum definition, layer property, network property를 정의하는 모듈이다. 예를 들어서 
  Convolution Layer의 경우에 stride, pad, kernel size 등의 값을 정의할 수 있어야 한다.
  그러한 값들을 관리해주는 모듈을 뜻한다. layer property는 각 layer마다의 설정값을 의미하고, 
  network property는 각 network 마다의 설정값을 의미한다. enum definition은 사용자가 정의한
  enumeration type을 의미한다.

* Layer Property 등록 방법
 layerPropDef.json 파일에 등록할 Prop 이름과 그것의 5가지 속성을 json format에 맞게 기입한다.

 [layerPropDef.json]
    :
"Conv" : 
{
    "DESC" : "convolution layer",
    "PARENT" : "Base",
    "LEVEL" : 1,
    "LEARN" : 2,
    "PROPDOWN" : [true],
    "VARS" : 
        [   
            ["deconv", "bool", "false"],
            ["deconvExtraCell", "int", "0"],
            ["filterDim", "filter_dim", "{1, 1, 1, 1, 0, 1}",
                [
                    ["rows", "uint32_t"], 
                    ["cols", "uint32_t"],
                    ["channels", "uint32_t"],
                    ["filters", "uint32_t"],
                    ["pad", "uint32_t"],
                    ["stride", "uint32_t"]
                ]
            ],
                     :
        ]
},
                     :


총 6개의 속성이 정의 된다. 설명은 아래와 같다:
(1) DESC : 해당 prop에 대한 설명이다. 영어로 작성한다.
(2) PARENT : 상속받고자 하는 prop을 기입한다. 상속을 받을 것이 없는 경우에는 빈문자열을 기입
            한다.
(3) LEVEL : 상속의 관계를 tree로 나타냈을때의 depth를 나타낸다. 가장 상위의 Base prop은 0 
           이라는 level값을 가진다. Base prop을 상속하는 Conv prop은 1이라는 level 값을
           가진다. 만약 Conv prop을 상속하는 ABC라는 prop이 있다면 ABC prop은 2 level 값을 
           가진다.
(4) LEARN : 학습 파라미터의 개수를 기입한다. 1이상이면 학습을 할 수 있는 레이어이다.
(5) PROPDOWN : default propDown 값을 설정한다. 사용자가 propDown값을 정의하지 않은 경우에
              여기서 설정된 default propDown 값을 갖게 된다. 이 값을 비어있는 array로 정의
              를 하는 경우에는 default propDown은 input 개수만큼 true가 설정이 되는 것과
              동일한 효과를 가지게 된다.
(6) VARS : prop에서 정의하는 여러가지 속성값들을 의미한다. VARS는 여러개의 VAR로 정의된다.
           일반적인 VAR은 (VAR의 이름, VAR의 타입, VAR의 초기값) 3가지 튜플로 정의된다.
           만약 초기값을 특정 헤더파일에 정의되어 있는 타입으로 정의하고 싶다면 해당
           헤더파일을 genProp.py의 headerFileList에 추가한다.
           VAR의 타입은 primitive 형(ex. bool, int, float, ...)과 std::vector, std::string을
           지원한다. 
           사용자 정의 구조체를 사용하는 경우에는 구조체 변수 정보에 대한 리스트가 추가된다.
           filterDim과 변수를 참고하길 바란다.

 [genLayerPropList.py]
        :
####################################### Modify here ##########################################
# if you want to use specific custom type, you should insert header file that the custom type 
# is defined into headerFileList.
headerFileList = ["EnumDef.h, KistiInputLayer.h"]
##############################################################################################
        :

* Network Property 등록 방법
 layer property 등록방법과 매우 유사하다. networkPropDef.json 파일을 열어서 필요한 property를
 추가하면 된다. layer property의 VARS에 추가하는 것과 동일하다.

* prop 리스트 생성방법
 genLayerPropList.py, genNetworkProp.py를 실행한다.

* enum definition 등록 방법
 enumDef.json파일 열어서 정의하고자 하는 enumeration type 이름과 enumeration들을 열거하면
 된다. 예를 들어서 ParamFillerType은 3가지 enum값(ParamFillerType::Constant,
   ParamFillerType::Xavier, ParamfillerType::Gaussian)을 가지고 싶다고 한다면 아래와 
 같이 수정을 하면 된다.

 [enumDef.json]
       :
    {
        "NAME" : "ParamFillerType",
        "ENUM" : ["Constant", "Xavier", "Gaussian"]
    },
           
* enum definition 생성 방법
 genEnumpy를 실행한다.
