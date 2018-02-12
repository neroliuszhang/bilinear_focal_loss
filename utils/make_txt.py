# -*- coding:UTF-8 -*- 
import os
import cv2
import numpy as np
import csv

image_train_txt="/home/goerlab/Bilinear-CNN-TensorFlow/dataset/fgvc-aircraft-2013b/data/images_variant_test.txt"
image_train_file=open(image_train_txt)
lines=image_train_file.readlines()
image_dir="/home/goerlab/Bilinear-CNN-TensorFlow/dataset/fgvc-aircraft-2013b/data/images/"

generate_train_txt_name="/home/goerlab/Bilinear-CNN-TensorFlow/train_test_small/images_test.txt"
generate_train_txt=open(generate_train_txt_name,"wb")
generate_train_txt_file=file(generate_train_txt_name,"r+")

label_name_list=[]
label_name_cnt=0
line_cnt=0

for line in lines:
	line_cnt+=1
	if line_cnt%10==0:
		print(line.split(" "))
		line_content=line.split(" ")
		#print(len(line_content))
		label_name=""
		if len(line_content)>1:
			for i in range(1,len(line_content)):
				label_name+=line_content[i]
		if label_name not in label_name_list:
			label_name_list.append(label_name)
			label_name_cnt+=1
		image_name=image_dir+str(line_content[0])+'.jpg'
		print("%s,%s,label=%d" %(image_name,label_name,label_name_cnt-1))
		new_line=image_name+" "+str(label_name_cnt-1)+'\n'
		generate_train_txt_file.write(new_line)
print("all label num:%d" %len(label_name_list))
print("all lines:%d" %(line_cnt))
