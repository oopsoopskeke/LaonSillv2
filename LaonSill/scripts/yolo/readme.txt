* 유의사항
 pascal_voc 전처리기 스크립트 파일은 https://github.com/nilboy/tensorflow-yolo.git 
프로젝트에서 가져 와서 수정해서 사용하고 있다. 해당 프로젝트에서 라이센스 관련된 설명을 
발견하지 못해서 신경을 쓰고 있지는 않다. 나중에 직접 구현해도 상관 없다.

* 사용방법 - pascal_voc
 (1) pascal_voc 파일을 받아서 적절한 곳에 위치시킨다.
   $ mkdir -p /data/VOCPascal
   $ cd /data/VOCPascal
   $ curl -O https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
   $ curl -O https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
   $ tar xf VOCtrainval_06-Nov-2007.tar
   $ tar xf VOCtest_06-Nov-2007.tar
 (2) preprocess_pascal_voc.py 파일을 수정
   - DATA_PATH와 OUTPUT_PATH를 자신의 환경에 맞게 수정
 (3) pascal_voc 전처리기 스크립트를 실행
   $ python ./preprocess_pascal_voc.py

* 사용방법 - ilsvrc
 (1) ILSVRC 1000 class 파일을 받는다. 
     (파일을 구하지 못하면 이문헌 혹은 김종헌 연구원에게 문의 바람)
 (2) 적절한 곳에 압축을 풀어준다.
   $ cd /data
   $ tar xf ilsvrc12_train.tar
 (3) class.txt파일을 ilsvrc 데이터 폴더에 복사
   $ cp $LAONSILL_HOME/scripts/yolo/class.txt /data/ilsvrc12_train/.
 (4) preprocess_ilsvrc.py 파일을 수정
   - ILSVRC_ROOT_PATH를 자신의 환경에 맞게 수정
 (5) preprocess_ilsvrc.py 파일을 실행
   $ python ./preprocess_ilsvrc.py

