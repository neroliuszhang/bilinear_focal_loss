# encoding=utf-8
import csv
import matplotlib.pyplot as plt
from pylab import *
import numpy as np


# csv_file="/home/goerlab/Welder_detection/data_record/20171114_Tue/cro-val3/unbalanced/vgg_16_ft+sensi+l2/result_detail2/detail_result.csv"


# margin=0
def calcute(csv_file, margin, writer):
    csv_reader = csv.reader(open(csv_file))

    total = csv_reader.line_num

    print("total:%d" % (total))

    prediction_OK = 0
    prediction_OK_wrong = 0

    prediction_NG = 0
    prediction_NG_wrong = 0

    prob_pred_NG_to_OK = 0

    label_OK = 0
    label_NG = 0

    count = 0

    for row in csv_reader:
        if csv_reader.line_num == 1:
            continue
        else:
            prediction = int(row[1])

            label = int(row[0])
            if label == 0:
                label_OK += 1
            else:
                label_NG += 1
            proba = float(row[2])
            if prediction == 0:
                if proba >= margin:
                    if label == 0:
                        prediction_OK += 1
                        # label_OK+=1
                    else:
                        prob_pred_NG_to_OK += float(row[2])
                        count += 1
                        prediction_OK += 1
                        prediction_OK_wrong += 1
                        # label_NG+=1
                else:
                    print(row)
                    writer.writerow([row])
                    prediction = 1
                    if label == 0:
                        prediction_NG += 1
                        prediction_NG_wrong += 1
                        # label_OK+=1
                    else:
                        prob_pred_NG_to_OK += float(row[2])
                        count += 1
                        prediction_NG += 1
                        # label_NG+=1
            else:
                if label == 0:
                    prediction_NG += 1
                    prediction_NG_wrong += 1
                    # label_OK+=1
                else:
                    prediction_NG += 1
                    # label_NG+=1
    if prediction_OK == 0:
        prediction_OK = 1
    # miss_rate=1.0*prediction_OK_wrong/prediction_OK

    # false_rate=1.0*prediction_NG_wrong/prediction_NG
    miss_rate = 1.0 * prediction_OK_wrong / label_NG
    false_rate = 1.0 * prediction_NG_wrong / label_OK

    if count == 0:
        count = 1
    average_prob = 1.0 * prob_pred_NG_to_OK / count
    print("margin:%f" % (margin))
    print("prediction_OK:%d" % (prediction_OK))
    print("prediction_OK_wrong:%d" % (prediction_OK_wrong))
    print("prediction_NG:%d" % (prediction_NG))
    print("prediction_NG_wrong:%d" % (prediction_NG_wrong))
    print("label_OK:%d" % (label_OK))
    print("label_NG: %d" % (label_NG))

    print("miss rate: %f" % (miss_rate))
    print("false rate: %f" % (false_rate))
    print("average-prob: %f" % (average_prob))

    writer.writerow(["margin:%f" % (margin)])
    writer.writerow(["label_OK:%d" % (label_OK)])
    writer.writerow(["prediction_OK_wrong:%d" % (prediction_OK_wrong)])
    writer.writerow(["label_NG: %d" % (label_NG)])
    writer.writerow(["prediction_NG_wrong:%d" % (prediction_NG_wrong)])

    writer.writerow(["miss rate: %f" % (miss_rate)])
    writer.writerow(["false rate: %f" % (false_rate)])
    writer.writerow(["average prob: %f" % (average_prob)])
    writer.writerow(["\n"])

    return miss_rate, false_rate, average_prob


if __name__ == "__main__":
    dirc_name = "/home/goerlab/EEA_8000/"
    csv_file = dirc_name + "/Together_all.csv"
    write_file = file(dirc_name + "/confusion_matrix_analysis.csv", "wb")
    # write_file="margin_analysis.csv"
    writer = csv.writer(write_file)
    margin = [0, 0.3, 0.4,0.5,0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 1]
    # margin = np.linspace(0,0.996,num=400)
    miss_rate_all = []

    false_rate_all = []

    for i in margin:
        print(i)
        miss_rate, false_rate, average_prob = calcute(csv_file, i, writer)
        miss_rate_all.append(miss_rate)
        false_rate_all.append(false_rate)

    print("miss_rate:")
    print(miss_rate_all)
    print("false_rate:")
    print(false_rate_all)
    writer.writerow(["miss rate: "])
    writer.writerow(miss_rate_all)
    writer.writerow(["false rate:"])
    writer.writerow(false_rate_all)

    dirc_name_2 = "/home/goerlab/EEA_8000/"
    csv_file_2 = dirc_name_2 + "Together_all.csv"
    write_file_2 = file(dirc_name_2 + "/margin_analysis.csv", "wb")
    # write_file="margin_analysis.csv"
    writer_2 = csv.writer(write_file_2)
    margin_2 = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.85,0.87,0.89, 0.9,0.92,0.95,0.97, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 1]
    # margin = np.linspace(0,0.996,num=400)
    miss_rate_all_2 = []

    false_rate_all_2 = []

    for i in margin_2:
        print(i)
        miss_rate_2, false_rate_2, average_prob_2 = calcute(csv_file_2, i, writer_2)
        miss_rate_all_2.append(miss_rate_2)
        false_rate_all_2.append(false_rate_2)

    print("miss_rate:")
    print(miss_rate_all_2)
    print("false_rate:")
    print(false_rate_all_2)
    writer.writerow(["miss rate: "])
    writer.writerow(miss_rate_all_2)
    writer.writerow(["false rate:"])
    writer.writerow(false_rate_all_2)

    plt.plot(false_rate_all, miss_rate_all, "r.-", label="VGG19_512input")
    plt.plot(false_rate_all_2, miss_rate_all_2, marker='.', mec='g', mfc='g',label="Bilimear-CNN")

    plt.xlabel('false-rate', fontproperties='SimHei')
    plt.ylabel('miss-rate', fontproperties='SimHei')

    legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
    #legend = plt.legend(loc='upper right', title='method1:', shadow=True, fontsize='small')
    # plt.legend(loc='upper right')
    # legend.get_title().set_fontsize(fontsize = 14)

    x = [0.1, 1.0, 0, 0.065]
    plt.axis(x)

    a=false_rate_all[0]
    b=miss_rate_all[0]
    #pl_a=float('%.2f' % a)
    a_p=float('%.4f' % a)
    b_p=float('%.4f' % b)
    ab=(a_p,b_p)

    a_2=false_rate_all_2[0]
    b_2=miss_rate_all_2[0]

    a_p2=float('%.4f' % a_2)
    b_p2=float('%.4f' % b_2)
    ab_2=(a_p2,b_p2)

    plt.text(a,b,str(ab))
    plt.text(a_2,b_2,str(ab_2))


    # dirc_name = "/home/goerlab/Welder_detection/data_record/20180110/"
    # plt.savefig(dirc_name + "/stage3_3.png")
    plt.show()