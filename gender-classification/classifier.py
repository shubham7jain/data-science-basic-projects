from sklearn import tree
from sklearn import naive_bayes
from sklearn import neighbors

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clfTree = tree.DecisionTreeClassifier()
clfTree = clfTree.fit(X, Y)

prediction = clfTree.predict([[190, 70, 43]])

print('Prediction from DecisionTreeClassifier is ', prediction)

clfNB = naive_bayes.GaussianNB()
clfNB = clfNB.fit(X, Y)

prediction = clfNB.predict([[190, 70, 43]])

print('Prediction from GaussianNBClassifier is ', prediction)

clfKN = neighbors.KNeighborsClassifier()
clfKN = clfKN.fit(X, Y)

prediction = clfKN.predict([[190, 70, 43]])

print('Prediction from KNeighborsClassifier is ', prediction)