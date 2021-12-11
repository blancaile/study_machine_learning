import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#暗号取得
f = open(os.getcwd() + "\decryption_homophonic_cipher\homophonic.txt","r")
data = f.read()
f.close()
print(type(data))


dfdata = data.split()

frequency = len(data.split())
print(frequency)



#頻度分布取得
hist_code1 = pd.DataFrame(columns=["count", "frequency"])

for i in range(32):
    if i < 10:
        #print("0" + str(i), ": ", data.count("0" + str(i)))
        hist_code1.loc[i] = [data.count("0" + str(i)), data.count("0" + str(i)) * 100 / frequency]
    else:
        #print(i, ": ", data.count(str(i)))
        hist_code1.loc[i] = [data.count(str(i)), data.count(str(i)) * 100 / frequency]
    

print(hist_code1["frequency"])


#連続した文字列の取得
# hist_code2 = pd.DataFrame(columns=["index", "count"])

# k = 0
# for i in range(32):
#   for j in range(32):
#        if i < 10 and j < 10:
#            hist_code2.loc[k] = ["0" + str(i) + " " + "0" + str(j), data.count("0" + str(i) + " " + "0" + str(j))]
#        elif i < 10 and j >= 10:
#            hist_code2.loc[k] = ["0" + str(i) + " " + str(j), data.count("0" + str(i) + " " + str(j))]
#        elif i >= 10 and j < 10:
#            hist_code2.loc[k] = [str(i) + " " + "0" + str(j), data.count(str(i) + " " + "0" + str(j))]
#        else:
#            hist_code2.loc[k] = [str(i) + " " + str(j), data.count(str(i) + " " + str(j))]
#        k+=1

# hist2 = hist_code2.sort_values("count", ascending=False)

# print(hist2.head(30))



# hist_code3 = pd.DataFrame(columns=["index", "count"])

# h = 0
# for i in range(32):
#     for j in range(32):
#         for k in range(32):
#             if i < 10 and j < 10 and k < 10:
#                 hist_code3.loc[h] = ["0" + str(i) + " " + "0" + str(j) + " " + "0" + str(k), data.count("0" + str(i) + " " + "0" + str(j) + " " + "0" + str(k))]
#             elif i < 10 and j < 10 and k >= 10:
#                 hist_code3.loc[h] = ["0" + str(i) + " " + "0" + str(j) + " " + str(k), data.count("0" + str(i) + " " + "0" + str(j) + " " + str(k))]

#             elif i < 10 and j >= 10 and k < 10:
#                 hist_code3.loc[h] = ["0" + str(i) + " " + str(j) + " " + "0" + str(k), data.count("0" + str(i) + " " + str(j) + " " + "0" + str(k))]
#             elif i < 10 and j >= 10 and k >= 10:
#                 hist_code3.loc[h] = ["0" + str(i) + " " + str(j) + " " + str(k), data.count("0" + str(i) + " " + str(j) + " " + str(k))]

#             elif i >= 10 and j < 10 and k < 10:
#                 hist_code3.loc[h] = [str(i) + " " + "0" + str(j) + " " + "0" + str(k), data.count(str(i) + " " + "0" + str(j) + " " + "0" + str(k))]
#             elif i >= 10 and j < 10 and k >= 10:
#                 hist_code3.loc[h] = [str(i) + " " + "0" + str(j) + " " + str(k), data.count(str(i) + " " + "0" + str(j) + " " + str(k))]

#             else:
#                 hist_code3.loc[h] = [str(i) + " " + str(j) + " " + str(k), data.count(str(i) + " " + str(j) + " " + str(k))]
#             h+=1
#             print("\r"+str(h)+" / "+str(32*32*32), end="")

# hist3 = hist_code3.sort_values("count", ascending=False)
# print(" ")
# print(hist3.head(30))



#      index count
    #312   12 28     8
    #914   28 18     7
    #1020  31 28     7
#679   21 07     7
#44    01 12     6
#405   12 21     5
#346   10 26     5
#970   30 10     5
#321   10 01     4
#922   28 26     4
#1013  31 21     4
#309   09 21     4
#696   21 24     4
#680   21 08     4
#154   04 26     4
#606   18 30     3
#39    01 07     3
#812   25 12     3
#287   08 31     3


#          index count
    #13185   12 28 1     4
    #32641   31 28 1     4
    #13202  12 28 18     4
#21056   20 18 0     3
#12674   12 12 2     3
#29278  28 18 30     3
#29506   28 26 2     3
#26370   25 24 2     3
#27264   26 20 0     3
#25986   25 12 2     3
#12961   12 21 1     3
#29251   28 18 3     3
#9895   09 21 07     3
    #32658  31 28 18     3
#32130   31 12 2     3
#25217   24 20 1     3
#29249   28 18 1     2
#26005  25 12 21     2
#14721   14 12 1     2
#23861  23 09 21     2


#print(hist_code2)
#print(hist_code2.shape)
#print(hist_code1["frequency"])

#hist_code1["count"].plot.bar()
#plt.show()


#for i in range(len(dfdata)):
#    if dfdata[i] == 28 and dfdata[i+1] == 18 or dfdata[i+1] == 01

print(dfdata[0])
print(type(dfdata[0])) #str


for i in range(len(dfdata)-2):
    if dfdata[i] == "28" and (dfdata[i+1] == "01" or dfdata[i+1] == "18") and (dfdata[i+2] == "01" or dfdata[i+2] == "18"):
        print(dfdata[i-2] + " " + dfdata[i-1] + " " + dfdata[i] + " " + dfdata[i+1] + " " + dfdata[i+2])

# 05 04 31 28 18
# 29 04 26 31 18
# 01 21 07 31 18
# 16 12 12 28 18
# 24 21 30 10 18
# 17 19 31 28 18
# 17 27 07 04 18
# 27 31 12 28 18
# 05 12 12 28 18
# 24 08 05 23 18
# 19 14 31 28 18
# 21 25 12 28 18
# 05 21 07 12 18
# 30 01 12 21 18
# 29 31 28 11 18
# 24 31 13 20 18
# 11 29 10 11 18
# 19 08 24 21 18
# 19 21 24 20 18
# 21 24 20 18 01
# 08 30 21 20 18
# 24 17 25 31 18
# 26 25 17 12 18
# 18 31 28 01 12
# 28 21 24 12 18
# 12 18 05 15 18
# 30 10 29 14 18
# 21 31 27 04 18


df = pd.DataFrame(np.arange(100))
print(df)

df = df.drop(i for i in range(len(df) - 5, len(df)))
print(df)