from decisionTreeClassifier import decisionTree
from MultiDecisionTree import MultiDecisionTree
from regression import regression


r = decisionTree('spambase.csv')
r.foldup()


r = decisionTree('mushroom.csv')
r.foldup()

r = decisionTree('iris.csv')
r.foldup()






r = MultiDecisionTree('mushroom.csv')
r.foldup()





r = regression('housing.csv',10,10)
r.foldup()