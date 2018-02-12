import csv
import shutil
import os

def GetFileNameAndExt(filename):
	(filepath,tempfilename) = os.path.split(filename);
	(shotname,extension) = os.path.splitext(tempfilename);
	return filepath,shotname,extension

csv_file="/home/goerlab/Bilinear-CNN-TensorFlow/20180126/Compact_Bilinear_CNN/model/last_layers_epoch_30_bilinear_focal_balance_0205_alpha-0.25_sensi1_1-1.npz/Image2001_20180201.csv"
csv_reader=csv.reader(open(csv_file))
total=csv_reader.line_num
dst_dir="/home/goerlab/Bilinear-CNN-TensorFlow/20180126/Compact_Bilinear_CNN/model/last_layers_epoch_30_bilinear_focal_balance_0205_alpha-0.25_sensi1_1-1.npz/"+"Image_2001/"
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

for row in csv_reader:
    if csv_reader.line_num==1:
        continue

    label=int(row[0])
    print("label:%d" %(label))
    predict=int(row[1])
    print("predict:%d" %(predict))
    proba=float(row[2])
    print("proba:%f" %(proba))
    file_name=row[3]
    #print("filename:%s" %(file_name))
    if predict==0 and proba>0.65 and label>0:
        print("here")
        path,name,ext=GetFileNameAndExt(file_name)
        #print(dst_dir + "/" + name + ext)
        shutil.copy(file_name,dst_dir+"/"+name+ext)


