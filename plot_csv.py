import csv
import matplotlib.pyplot as plt
from IPython import display

csv_files = ['runs/weights/yolov5s/results.csv',
             'runs/weights/yolov5s-at/results.csv',
             'runs/weights/yolov5s-at-conf-kd/results.csv',
             'runs/weights/yolov5s-ca/results.csv',
             'runs/weights/yolov5s-ca-at/results.csv']
label = ['yolov5s',
         'yolov5m--->yolov5s',
         'yolov5m-se--->yolov5s',
         'yolov5m-cbam--->yolov5s',
         'yolov5m-ca--->yolov5s']
color = ['m', 'c', 'r', 'b', 'k']
ls = [':', '--', '-.', ':', '-.']
marker = ['1', ',', ',', '*', ',']

plt.figure(1)
for j in range(len(csv_files)):
    exampleFile = open(csv_files[j])
    exampleReader = csv.reader(exampleFile)
    exampleData = list(exampleReader)
    length_zu = len(exampleData)

    x = list()
    y = list()

    for i in range(1, length_zu):
        x.append(float(exampleData[i][0]))
        y.append(float(exampleData[i][6])*100)

    plt.plot(x, y, ls=ls[j], color='k', marker=marker[j], markersize=2, lw=1, label=label[j])

plt.xlabel('Epoch')
plt.ylabel('mAP/%')
plt.legend(loc='lower right', fontsize=8)

plt.savefig("kd.png", dpi=600, format='png')
plt.show()
