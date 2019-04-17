import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import operator
from sklearn.metrics import accuracy_score, precision_score, recall_score

'''
#getting data

train_data = open('train_data.csv')


df = pd.read_csv(train_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])
docIdx = df['docIdx'].values




train_label = open('train_label.csv')
label = []
lines = train_label.readlines()
for line in lines:
    label.append(int(line.split()[0]))
i = 0
new_label = []
for index in range(len(docIdx)-1):
    new_label.append(label[i])
    if docIdx[index] != docIdx[index+1]:
        i += 1
new_label.append(label[i])
df['classIdx'] = new_label


#PriorProbility
train_label = open('train_label.csv')
p = pd.read_csv(train_label, delimiter=' ', names=['classIdx'])
t = p.groupby('classIdx').size().sum()
Prior_P = (p.groupby('classIdx').size())/t


#select vocabalury
words = df.groupby('wordIdx').sum().sort_values('count',ascending=False)['count']
print(words)
t=80000
if t in words.index:
    print('ok')
else :print('bad')




#traindata

totalDocs = df.groupby('docIdx').count().shape[0]

pb_ij = df.groupby(['classIdx','wordIdx'])
pb_j = df.groupby(['classIdx'])


Pr_MB =  (pb_ij['count'].count() + 1) / (pb_j['count'].count() + totalDocs)
Pr_MB = Pr_MB.unstack()
for c in range(1,21):
    Pr_MB.loc[c,:] = Pr_MB.loc[c,:].fillna(1/(pb_j['count'].count()[c] + totalDocs))
Pr_MB = np.log(Pr_MB)

Pr_MB_dict = Pr_MB.to_dict()


#Total words need to be adjusted to vocabalury
totalWords = df.groupby('wordIdx').count().shape[0]
pb_ij = df.groupby(['classIdx','wordIdx'])
pb_j = df.groupby(['classIdx'])
Pr_MN =  (pb_ij['count'].sum() + 1) / (pb_j['count'].sum() + totalWords)
Pr_MN = Pr_MN.unstack()
for c in range(1,21):
    Pr_MN.loc[c,:] = Pr_MN.loc[c,:].fillna(1/(pb_j['count'].sum()[c] + totalWords))
Pr_MN = np.log(Pr_MN)
Pr_MN_dict = Pr_MN.to_dict()


#predict test data


test_data = open('test_data.csv')
test_label = open('test_label.csv')

df = pd.read_csv(test_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])

def MNB(df,Pr_dict,pi):

    # Using dictionaries for greater speed
    df_dict = df.to_dict()
    new_dict = {}
    prediction = []

    # new_dict = {docIdx : {wordIdx: count},....}
    for idx in range(len(df_dict['docIdx'])):
        docIdx = df_dict['docIdx'][idx]
        wordIdx = df_dict['wordIdx'][idx]
        count = df_dict['count'][idx]
        try:
            new_dict[docIdx][wordIdx] = count
        except:
            new_dict[df_dict['docIdx'][idx]] = {}
            new_dict[docIdx][wordIdx] = count

    # Calculating the scores for each doc
    for docIdx in range(1, len(new_dict) + 1):
        score_dict = {}
        # Creating a probability row for each class
        for classIdx in range(1, 21):
            score_dict[classIdx] = 0
            # For each word:
            for wordIdx in new_dict[docIdx]:

                try:
                    probability = Pr_dict[wordIdx][classIdx]
                    power = np.log(1 + new_dict[docIdx][wordIdx])

                    score_dict[classIdx] += power * probability
                except:
                    score_dict[classIdx] += 0
                    # f*log(Pr(i|j))

            score_dict[classIdx] += np.log(pi[classIdx])

            # Get class with max probabilty for the given docIdx
        max_score = max(score_dict, key=score_dict.get)
        prediction.append(max_score)

    return prediction

def calculateClassMetrics(testLabel, prediction):
    testClasses = [int(s) for s in testLabel.read().split()]

    precision = precision_score(
        testClasses, prediction, average=None)
    recall = recall_score(testClasses, prediction, average=None)

    print("Precision::\n{}".format(precision))
    print("Recall::\n{}".format(recall))

prediction = MNB(df,Pr_MB_dict,Prior_P)
calculateClassMetrics(test_label,prediction)

df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=df.values,
         rowLabels=df.index,
         colLabels=df.columns,
         cellLoc='center', rowLoc='center',
         loc='center')
fig.tight_layout()
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

x =list(range(len(voc)))
width=0.35
plt.bar(x, aa, width=width, label='MultiB',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, aaa, width=width, label='Multi',tick_label = voc,fc = 'r')
plt.legend()
plt.show()



plt.bar(x, pp, width=width, label='MultiB',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, ppp, width=width, label='Multi',tick_label = voc,fc = 'r')
plt.legend()
plt.show()


plt.bar(x, rr, width=width, label='MultiB',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, rrr, width=width, label='Multi',tick_label = voc,fc = 'r')
plt.legend()
plt.show()


'''
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
cm = confusion_matrix(testClasses, prediction)
# array([[1, 0, 0],
#   [1, 0, 0],
#   [0, 1, 2]])

# Now the normalize the diagonal entries
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# array([[1.        , 0.        , 0.        ],
#      [1.        , 0.        , 0.        ],
#      [0.        , 0.33333333, 0.66666667]])

# The diagonal entries are the accuracies of each class
cm.diagonal()
'''