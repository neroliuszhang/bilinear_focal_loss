from tflearn.data_utils import build_hdf5_image_dataset
import h5py

#new_train = "/home/adoke/tf_tutorial/aircrafts_new/new_train_val/new_train.txt"
#new_val = "/home/adoke/tf_tutorial/aircrafts_new/new_train_val/new_val.txt"
#new_test = "/home/adoke/tf_tutorial/aircrafts_new/from_start/a3_variants_test.txt"
new_train="/media/goerlab/My Passport/Welder_detection/dataset/20180209/h5_middle_crop/train2.txt"
new_val="/media/goerlab/My Passport/Welder_detection/dataset/20180209/h5_middle_crop/val.txt"
#new_test="/home/goerlab/Bilinear-CNN-TensorFlow/train_test_small/images_test.txt"

dir="/media/goerlab/My Passport/Welder_detection/dataset/20180209/h5_middle_crop/"

#image_shape option can be set to different values to create images of different sizes
# build_hdf5_image_dataset(new_val, image_shape=(448, 448), mode='file', output_path=dir+'new_val_448.h5', categorical_labels=True, normalize=False)
#
# #
# print 'Done creating new_val.h5'
#build_hdf5_image_dataset(new_test, image_shape=(448, 448), mode='file', output_path=dir+'new_test_224.h5', categorical_labels=True, normalize=False)
#print 'Done creating new_test.h5'
build_hdf5_image_dataset(new_train, image_shape=(448, 448), mode='file', output_path=dir+'new_train_448.h5', categorical_labels=True, normalize=False)
print 'Done creating new_train_488.h5'

