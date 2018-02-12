# -*- coding:UTF-8 -*-
import os
import cv2
import numpy as np
import csv
import random

image_dir="/media/goerlab/My Passport/Welder_detection/dataset/20180209/val_crop/"
record_name="/media/goerlab/My Passport/Welder_detection/dataset/20180209/h5_middle_crop/val.txt"
image_record=open(record_name,"wb")
image_record_file=file(record_name,"r+")

for dirc in os.listdir(image_dir):
    sub_dir=image_dir+dirc+'/'
    #print(sub_dir)
    for sub_dirc in os.listdir(sub_dir):
        file_name=sub_dir+sub_dirc
        #print(file_name)
        if "Good" in file_name:
            real_label = 0
        elif "NoWeld" in file_name:
            real_label = 1
        elif "NoWire" in file_name:
            real_label = 2
        elif "ExtraWire_single" in file_name or "extrawire" in file_name:
            real_label = 3
        elif "ExtraWire_double" in file_name:
            real_label = 4
        elif "Offset" in file_name or "offset" in file_name:
            real_label = 5
        elif "WireGap" in file_name:
            real_label = 6
        else:
            real_label = 7

        line=file_name+" "+str(real_label)+"\n"
        print(line)
        image_record_file.write(line)

print("done")

# txt_name1="/media/goerlab/My Passport/Welder_detection/dataset/20180209/h5_middle_crop/train.txt"
# txt_name2="/media/goerlab/My Passport/Welder_detection/dataset/20180209/h5_middle_crop/train2.txt"
# txt_file1=open(txt_name1,"r")
# rows=txt_file1.readlines()
# txt_file1.close()
# random.shuffle(rows)
# #
# txt_file2=open(txt_name2,"w")
# txt_file2.writelines(rows)
# txt_file2.close()

# image_train_txt="/home/goerlab/Bilinear-CNN-TensorFlow/dataset/fgvc-aircraft-2013b/data/images_variant_test.txt"
# image_train_file=open(image_train_txt)
# lines=image_train_file.readlines()
# image_dir="/home/goerlab/Bilinear-CNN-TensorFlow/dataset/fgvc-aircraft-2013b/data/images/"
#
# generate_train_txt_name="/home/goerlab/Bilinear-CNN-TensorFlow/train_test_small/images_test.txt"
# generate_train_txt=open(generate_train_txt_name,"wb")
# generate_train_txt_file=file(generate_train_txt_name,"r+")
#
# label_name_list=[]
# label_name_cnt=0
# line_cnt=0
#
# for line in lines:
# 	line_cnt+=1
# 	if line_cnt%10==0:
# 		print(line.split(" "))
# 		line_content=line.split(" ")
# 		#print(len(line_content))
# 		label_name=""
# 		if len(line_content)>1:
# 			for i in range(1,len(line_content)):
# 				label_name+=line_content[i]
# 		if label_name not in label_name_list:
# 			label_name_list.append(label_name)
# 			label_name_cnt+=1
# 		image_name=image_dir+str(line_content[0])+'.jpg'
# 		print("%s,%s,label=%d" %(image_name,label_name,label_name_cnt-1))
# 		new_line=image_name+" "+str(label_name_cnt-1)+'\n'
# 		generate_train_txt_file.write(new_line)
# print("all label num:%d" %len(label_name_list))
# print("all lines:%d" %(line_cnt))
