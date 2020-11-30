* 사용방법
 (1) yolo 관련 전처리 스크립트(scripts/yolo/preprocess_pascal_voc.py)를 실행한다. 사용 방법은
  yolo 전처리 스크립트의 readme.txt(scripts/yolo/readme.txt) 파일을 참고한다.

 (2) preprocess.py파일을 열어서 sourceMetaFilePath와 만들어질 targetMetaFilePath를 적절히
  설정한다. sourceMetaFilePath는 (1) 단계의 결과물을 가리켜야 한다.

 (3) conver_imageset 툴을 이용하여 sdf 포맷으로 변경한다. 아래와 같이 bash script를 사용하여 변경하였다.

  [convert.sh]
#!/bin/sh
vocpascal_image_path='/'                                                                          
vocpascal_dataset_path='/data/VOCdevkit/VOCdevkit/pascal_voc_multilabel_only.txt'                 
vocpascal_sdf_path='/data/sdf/pascalvoc_multilabel/'                                              
                                                                                                  
echo "delete $vocpascal_sdf_path"                                                                 
rm -rf $vocpascal_sdf_path                                                                        
                                                                                                  
echo "converting ... "                                                                            
convert_imageset -m -i $vocpascal_image_path -d $vocpascal_dataset_path -o $vocpascal_sdf_path   

 $ ./convert.sh
