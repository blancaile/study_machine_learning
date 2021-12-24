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
    

#print(hist_code1["frequency"])


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
# #print(hist3.head(50))
# hist3.to_csv(os.getcwd() + "\decryption_homophonic_cipher/three_text.csv")

def trans(i):
    if i == "12" or i == "31":
        i = "T "
    elif i == "28":
        i = "H "
    elif i == "18":
        i = "E "        #ここまでは合ってる前提

    return i
    


def decryption(df, t):
    df = df.query("count >= "+str(t))
    for j in df.itertuples():
        for i in j[2].split():
            i = trans(i)
            print(i, end=" ")#index
        print("  " + str(j[3]))#index
        #print(i[3])#count
        

twotext = pd.read_csv(os.getcwd() + r"\decryption_homophonic_cipher/two_text.csv")
threetext = pd.read_csv(os.getcwd() + r"\decryption_homophonic_cipher/three_text.csv")

decryption(twotext, 2)
decryption(threetext, 2)

#暗号文の出力
for i in dfdata:
    i = trans(i)
    print(str(i),end=" ")

#12 28
#31 28 をTHと仮定 <-一番多く出ているから
    #12, 31 T
    #28     H
    #{18, 26, 05}のうち２個がEになる
    #28 18 30 をHERと仮定 <-最初がHで３文字の頻出文字列だから
        #18     E
        #30     R
        #{05,26}のうち１個がEになる

#26 05 08
#26 20 08 をINGと仮定 <-１番目と３番目が同じで３文字の頻出文字列だから
    #26     I
    #05, 20 N
    #08     G
    #24をSと仮定 <-28 26 24 のうちHIまで分かっていてHISが３文字の頻出文字列だから
        #24 S







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