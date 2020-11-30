#!/bin/sh

if [ "$#" -ne 1 ]
then
	echo "invalid usage: $0 type"
	exit 0
fi

base_image_path=$LAONSILL_HOME'/data/'
base_sdf_path=$LAONSILL_HOME'/data/sdf/'


if [ "$1" = "celeba" ]
then
	celeba_image_path=$base_image_path'celebA_small/'
	celeba_sdf_path=$base_sdf_path'celeba_sdf/'

	echo "delete $celeba_sdf_path"
	rm -rf $celeba_sdf_path

	echo "converting ... "
	convert_imageset -i $celeba_image_path -o $celeba_sdf_path 

elif [ "$1" = "mnist" ]
then
	mnist_image_path_train=$base_image_path'mnist/train-images-idx3-ubyte'
	mnist_label_path_train=$base_image_path'mnist/train-labels-idx1-ubyte'
	mnist_sdf_path_train=$base_sdf_path'mnist_train_sdf/'

	echo "delete $mnist_sdf_path_train"
	rm -rf $mnist_sdf_path_train

	echo "converting $0 train ... "
	convert_mnist_data -i $mnist_image_path_train -l $mnist_label_path_train -o $mnist_sdf_path_train

	mnist_image_path_test=$base_image_path'mnist/t10k-images-idx3-ubyte'
	mnist_label_path_test=$base_image_path'mnist/t10k-labels-idx1-ubyte'
	mnist_sdf_path_test=$base_sdf_path'mnist_test_sdf/'

	echo "delete $mnist_sdf_path_test"
	rm -rf $mnist_sdf_path_test

	echo "converting $0 ... "
	convert_mnist_data -i $mnist_image_path_test -l $mnist_label_path_test -o $mnist_sdf_path_test

elif [ "$1" = "ilsvrc" ]
then
	ilsvrc_base_path=$base_image_path'ilsvrc12_train/'
	ilsvrc_image_path=$ilsvrc_base_path'images/'
	ilsvrc_dataset_path=$ilsvrc_base_path'train_10000.txt'
	ilsvrc_sdf_path=$base_sdf_path'ilsvrc_10000_sdf/'

	echo "delete $ilsvrc_sdf_path"
	rm -rf $ilsvrc_sdf_path

	echo "converting ... "
	convert_imageset -s -w 224 -h 224 -i $ilsvrc_image_path -d $ilsvrc_dataset_path -o $ilsvrc_sdf_path 

elif [ "$1" = "esp" ]
then
	esp_base_path=$base_image_path'ESP-ImageSet/'
	esp_image_path=$esp_base_path'images/'
	esp_dataset_path=$esp_base_path'train_10000.txt'
	esp_sdf_path=$base_sdf_path'esp_10000_sdf/'

	echo "delete $esp_sdf_path"
	rm -rf $esp_sdf_path

	echo "converting ... "
	convert_imageset -m -i $esp_image_path -d $esp_dataset_path -o $esp_sdf_path 

elif [ "$1" = "kisti" ]
then
	kisti_base_path=$base_image_path'flatten_test/'
	kisti_image_path=$kisti_base_path''
	kisti_dataset_path=$kisti_base_path'train.txt'
	kisti_sdf_path=$base_sdf_path'kisti_10000_sdf/'

	echo "delete $kisti_sdf_path"
	rm -rf $kisti_sdf_path

	echo "converting ... "
	convert_imageset -m -i $kisti_image_path -d $kisti_dataset_path -o $kisti_sdf_path 
else
	echo "invalid data type: $1 ... "
	exit 0
fi

























