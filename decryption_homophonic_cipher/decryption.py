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

# hist_code3 = pd.DataFrame(columns=["index", "count"])
# add0 = lambda i: "0"+str(i) if i < 10 else str(i)

# h = 0
# for i in range(32):
#     for j in range(32):
#         for k in range(32):
#                 hist_code3.loc[h] = [add0(i) + " " + add0(j) + " " + add0(k), data.count(add0(i) + " " + add0(j) + " " + add0(k))]
#                 h+=1
#                 print("\r"+str(h)+" / "+str(32**3), end="")

# hist3 = hist_code3.sort_values("count", ascending=False).reset_index(drop=True)
# print(" ")
# print(hist3.head(50))


#暗号文の出力
for i in dfdata:
    #f i == "01" or i == "18":
    #    i = "E"
    #elif i == "03" or i == "30":
    #    i = "R"
    if i == "12" or i == "31":  #THE
        i = "T"
    elif i == "28":
        i = "H"
    elif i == "18": #HER
        i = "E"
    elif i == "30":
        i = "R"

    elif i == "24": #HIS
        i = "S"
    elif i == "26":
        i = "I"

    elif i == "05" or i == "20": #ING
        i = "N"
    elif i == "08":
        i = "G"

    elif i == "00": #AND
        i = "A"
    elif i == "01":
        i = "D"
    elif i == "26": #消去法でEが求まる
        i = "E"

    print(str(i),end=" ")



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


#        index count
# 0   12 28 18     4    THE
# 1   09 21 07     3    
# 2   31 28 18     3    THE
# 3   28 18 30     3    HER
# 4   31 18 16     2    TE
# 5   08 06 17     2    G
# 6   12 01 07     2    TD
# 7   00 20 01     2    AND      =>AND?
# 8   31 12 28     2    TTH
# 9   25 24 26     2     SI
# 10  18 16 31     2    E T
# 11  21 07 12     2      T
# 12  12 12 28     2    TTH
# 13  14 30 12     2     RT
# 14  26 20 08     2    ING
# 15  12 21 19     2    T
# 16  28 26 24     2    HIS       ->HIS?
# 17  19 17 23     2
# 18  23 09 21     2
# 19  24 21 30     2    S R
# 20  10 01 12     2     DT
# 21  08 14 12     2    G T
# 22  12 28 05     2    THN
# 23  31 28 26     2    THE
# 24  05 25 04     2    N
# 25  08 08 06     2    GG
# 26  25 12 21     2     T
# 27  01 12 21     2    DT
# 28  27 10 01     2      D
# 29  05 08 14     2    NG
# 30  26 05 08     2    ING
# 31  14 12 17     2     T
# 32  21 18 25     2      E


#           index count
# 0   05 08 14 12     2    T
# 1   26 05 08 14     2
# 2   23 09 21 07     2
# 3   12 12 28 18     2 TTHE
# 4   08 14 12 17     2   T
# 5   31 28 18 30     2 THER
# 6   08 08 06 17     2 
# 7   25 12 21 19     2  T
# 8   26 25 17 12     1
# 9   26 04 30 26     1
# 10  19 05 24 31     1
# 11  07 08 30 21     1
# 12  29 04 26 31     1
# 13  24 26 01 07     1
# 14  30 27 10 01     1
# 15  26 12 24 28     1
# 16  20 01 30 28     1
# 17  01 15 21 16     1
# 18  03 10 26 08     1
# 19  20 08 21 25     1
# 20  04 13 27 31     1
# 21  18 25 12 21     1
# 22  12 24 28 10     1
# 23  01 12 12 26     1
# 24  25 31 18 16     1
# 25  17 30 10 26     1
# 26  23 18 19 10     1
# 27  31 19 17 23     1
# 28  28 18 13 05     1
# 29  07 24 10 27     1



# for i in range(len(dfdata)-2):
#     if dfdata[i] == "28" and (dfdata[i+1] == "01" or dfdata[i+1] == "18") and (dfdata[i+2] == "01" or dfdata[i+2] == "18"):
#         print(dfdata[i-2] + " " + dfdata[i-1] + " " + dfdata[i] + " " + dfdata[i+1] + " " + dfdata[i+2])

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