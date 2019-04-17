import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import operator
from sklearn.metrics import accuracy_score, precision_score, recall_score
from MultiB import MultiB
from MultiNomial import MultiNomial
from Logistic import Logistic

p=Logistic('spambase.csv',0.0001,0.05)
p.foldup()



q = Logistic('breastcancer.csv',0.0001,0.05)
q.foldup()


t = Logistic('diabetes.csv',0.00001,0.01)
t.foldup()

multivariateBernoulli = MultiB()
pp,rr,aa,total,pb,rb,ab = multivariateBernoulli.foldup('train_data.csv', 'train_label.csv', 'test_data.csv','test_label.csv')



multinomial = MultiNomial()
p,r,a,total,pn,rn,an = multinomial.foldup('train_data.csv', 'train_label.csv', 'test_data.csv', 'test_label.csv')
voc = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, total]



x =list(range(len(voc)))
width=0.35
plt.bar(x, aa, width=width, label='MultiB',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, a, width=width, label='Multi',tick_label = voc,fc = 'r')
plt.legend()
plt.show()



plt.bar(x, pp, width=width, label='MultiB',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, p, width=width, label='Multi',tick_label = voc,fc = 'r')
plt.legend()
plt.show()


plt.bar(x, rr, width=width, label='MultiB',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, r, width=width, label='Multi',tick_label = voc,fc = 'r')
plt.legend()
plt.show()





classes=list(range(1,21,1))
x =list(range(len(classes)))
width=0.2
plt.bar(x, ab, width=width, label='MultiB',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, an, width=width, label='MultiN',tick_label = classes,fc = 'r')
plt.legend()
plt.show()



plt.bar(x, pb, width=width, label='MultiB',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, pn, width=width, label='MultiN',tick_label = classes,fc = 'r')
plt.legend()
plt.show()


plt.bar(x, rb, width=width, label='MultiB',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, rn, width=width, label='MultiN',tick_label = classes,fc = 'r')
plt.legend()
plt.show()
'''

pp= [0.06275816122584943, 0.6414390406395736, 0.7741505662891406, 0.7709526982011992, 0.7231179213857428, 0.7578947368421053, 0.7705529646902065, 0.7902731512325116, 0.7937375083277814, 0.7930712858094604, 0.7956029313790806]
rr =[0.05200201809942156, 0.6935063085287908, 0.785619823216475, 0.782959021383258, 0.7464147048829615, 0.7713125679468994, 0.7828346094851122, 0.8006826451688704, 0.8038283485966117, 0.8032666899271669, 0.805860953613858]
aa =[0.06275816122584943, 0.6414390406395736, 0.7741505662891406, 0.7709526982011992, 0.7231179213857428, 0.7578947368421053, 0.7705529646902065, 0.7902731512325116, 0.7937375083277814, 0.7930712858094604, 0.7956029313790806]
total = 53975
pb = [0.64615385, 0.6779661 , 0.703125 ,  0.60928433, 0.69933185, 0.878125,
 0.82984293, 0.81162791 ,0.91770574, 0.90640394, 0.96391753, 0.90053763,
 0.74184783, 0.94314381, 0.8953168 , 0.81235698, 0.69650655, 0.96989967,
 0.74248927, 0.66836735]
rb = [0.79245283, 0.71979434, 0.69053708, 0.80357143, 0.81984334, 0.72051282,
 0.82984293, 0.8835443,  0.92695214, 0.92695214 ,0.93734336, 0.84810127,
 0.69465649, 0.71755725, 0.82908163, 0.8919598,  0.87637363, 0.7712766,
 0.55806452, 0.52191235]
ab = [0.64615385, 0.6779661,  0.703125 ,  0.60928433, 0.69933185, 0.878125,
 0.82984293 ,0.81162791, 0.91770574 ,0.90640394, 0.96391753 ,0.90053763,
 0.74184783 ,0.94314381, 0.8953168 , 0.81235698 ,0.69650655, 0.96989967,
 0.74248927, 0.66836735]




ppp =  [0.07155229846768821, 0.6618254497001999, 0.7718854097268487, 0.7701532311792139, 0.7355096602265156, 0.7602931379080613, 0.7757495003331113, 0.7945369753497669, 0.798800799467022, 0.7914723517654897, 0.792271818787475]
rrr =  [0.1514441104952291, 0.6973886111192343, 0.7870292403377325, 0.7844465356169026, 0.7526908749534609, 0.773445307987764, 0.7875098086232297, 0.8037661259929275, 0.8072922367305333, 0.8025134708663951, 0.8037077666540821]
aaa =  [0.07155229846768821, 0.6618254497001999, 0.7718854097268487, 0.7701532311792139, 0.7355096602265156, 0.7602931379080613, 0.7757495003331113, 0.7945369753497669, 0.798800799467022, 0.7914723517654897, 0.792271818787475]

pbb= [0.70056497, 0.67035398, 0.81102362, 0.61277445, 0.80229226, 0.82162162,
 0.91272727, 0.79111111, 0.93246753, 0.96476965, 0.94320988, 0.77419355,
 0.78507463, 0.88918919, 0.87022901, 0.7164751,  0.67775468, 0.9039548,
 0.6,        0.8362069 ]
rbb= [0.77987421, 0.77892031, 0.52685422, 0.78316327, 0.7310705,  0.77948718,
 0.65706806, 0.90126582 ,0.90428212, 0.89672544 ,0.95739348 ,0.91139241,
 0.6692112 , 0.83715013, 0.87244898, 0.93969849, 0.8956044 , 0.85106383,
 0.59032258, 0.38645418]
abb= [0.70056497 ,0.67035398, 0.81102362 ,0.61277445 ,0.80229226, 0.82162162,
 0.91272727, 0.79111111, 0.93246753 ,0.96476965, 0.94320988, 0.77419355,
 0.78507463, 0.88918919, 0.87022901, 0.7164751 , 0.67775468, 0.9039548,
 0.6 ,       0.8362069 ]


voc = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, total]
'''