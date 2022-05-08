import csv
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("font", family='FangSong')
from IPython import display

csv_files = ['runs/student.csv',
             'runs/fps.csv',
             'runs/AT.csv',
             'runs/cd.csv',
             'runs/FFI.csv',
             'runs/ours1.csv']
label = ['学生网络',
         'FM-NMS',
         'AT',
         'CD',
         'FFI',
         '本文方法']
color = ['m', 'c', 'r', 'b', 'k']
ls = [':', '--', '-.', ':', '-.', '-']
marker = ['', '1', '', 'x', '4', '']

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
        y.append(float(exampleData[i][6]) * 100)

    plt.plot(x, y, ls=ls[j], color='k', marker=marker[j], markersize=1, lw=1, label=label[j])

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('mAP/%', fontsize=12)
plt.legend(loc='lower right', fontsize=14)

plt.savefig("kd.png", dpi=600, format='png')
plt.show()
