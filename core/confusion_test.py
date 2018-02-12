

confusion_matrix=[[281, 1, 0, 1, 5, 0, 2, 0, 0, 13, 0, 1, 0, 3],
[0, 16, 0, 0, 0, 0, 0, 0, 1, 0, 0, 6, 1, 0],
[1, 0, 40, 1, 1, 2, 0, 0, 0, 0, 0, 1, 0, 1],
[2, 2, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[1, 1, 3, 0, 66, 3, 3, 2, 0, 1, 0, 1, 0, 15],
[10, 5, 14, 5, 18, 224, 20, 9, 0, 8, 0, 11, 7, 2],
[0, 2, 0, 1, 3, 1, 189, 3, 0, 1, 0, 1, 2, 0],
[0, 0, 1, 0, 0, 0, 0, 7, 0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 0, 0],
[44, 2, 11, 14, 7, 18, 8, 2, 0, 132, 0, 6, 14, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0],
[0, 6, 0, 0, 2, 0, 0, 0, 0, 2, 0, 43, 0, 0],
[0, 2, 0, 1, 0, 1, 2, 0, 0, 5, 0, 0, 90, 0],
[2, 0, 0, 0, 29, 0, 2, 0, 0, 1, 0, 1, 0, 39]



]
num_of_class=13
print("confusion matrix:")
#print(confusion_matrix)
for i in range(num_of_class):
    print(confusion_matrix[i])
precision_result = [0 for i in range(num_of_class)]
recall_result = [0 for i in range(num_of_class)]
# for i in range(num_of_class):
#       precision_result[i]=confusion_matrix[i][i]/
precision_sum = map(sum, zip(*confusion_matrix))

# print("precision_sum:")
# print(precision_sum)
for i in range(num_of_class):
    if precision_sum[i]==0:
        precision_sum[i]=1
    precision_result[i] = 1.0*confusion_matrix[i][i] / precision_sum[i]

print("average_precision:")
print(precision_result)
print("mean_average_precision:")
print(sum(precision_result)/num_of_class)

# print("mean_average_precision:")
# print(sum(precision_result)/num_of_class)

#print("recall_sum:")
recall_sum = map(sum, confusion_matrix)
#print(recall_sum)

for i in range(num_of_class):
    if recall_sum[i]==0:
        recall_sum[i]=1
    recall_result[i] = 1.0*confusion_matrix[i][i] / recall_sum[i]

print("recall:")
print(recall_result)
print("mean_recall:")
print(sum(recall_result)/num_of_class)
real_accuracy=0
for i in range(num_of_class):

    real_accuracy+=confusion_matrix[i][i]

accuracy_final=1.0*real_accuracy/sum(recall_sum)
print(real_accuracy)
print(sum(recall_sum))
print(accuracy_final)
print("accuracy:%d/%d = %f" %(real_accuracy,sum(recall_sum),accuracy_final))