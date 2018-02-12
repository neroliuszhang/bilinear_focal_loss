import numpy as np

a=[90,90,80,40,30,80,90]
b=[10,80,70,60,40,30,20]
array_a=np.array(a)
max_a=np.argmax(array_a,axis=0)
#print(len(a))
print("a:")
print(array_a)
print(max_a)

array_b=np.asarray(b)
max_b=np.argmax(array_b)
print("b:")
#print(len(b))
print(max_b)

print("c:")
c=np.arange(6).reshape(2,3)
print(c)
print(np.argmax(c))


print("d:")
d=np.arange(6)
d[1]=5
print(d)
print(np.argmax(d))
# def to_categorical(y, nb_classes):
#     """ to_categorical.

#     Convert class vector (integers from 0 to nb_classes)
#     to binary class matrix, for use with categorical_crossentropy.

#     Arguments:
#         y: `array`. Class vector to convert.
#         nb_classes: `int`. Total number of classes.

#     """
#     y = np.asarray(y, dtype='int32')
#     if not nb_classes:
#         nb_classes = np.max(y)+1
#     Y = np.zeros((len(y), nb_classes))
#     Y[np.arange(len(y)),y] = 1.
#     return Y

# out=to_categorical([2],3)
# print(out)