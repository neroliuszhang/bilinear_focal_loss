import random
import os
txt_name1="/media/goerlab/My Passport/Welder_detection/download_everyday/20180123_VIDI/cross_validation_add_gan/cross1/train.txt"
txt_name2="/media/goerlab/My Passport/Welder_detection/download_everyday/20180123_VIDI/cross_validation_add_gan/cross1/train2.txt"

txt_file1=open(txt_name1,"r")
rows=txt_file1.readlines()
txt_file1.close()
random.shuffle(rows)

txt_file2=open(txt_name2,"w")
txt_file2.writelines(rows)
txt_file2.close()
