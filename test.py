from numpy import genfromtxt, zeros
# read the first 4 columns
data = genfromtxt('iris.csv', delimiter=',', usecols=(0,1,2,3))
# read the fifth column
target = genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)

t = zeros(len(target))
t[target == 'setosa'] = 1
t[target == 'versicolor'] = 2
t[target == 'virginica'] = 3

from numpy import corrcoef
corr = corrcoef(data.T) # .T gives the transpose

from pylab import pcolor, colorbar, xticks, yticks,show,plot
from numpy import arange
pcolor(corr)
colorbar() # add
# arranging the names of the variables on the axis
xticks(arange(0.5,4.5),['sepal length',  'sepal width', 'petal length', 'petal width'],rotation=-20)
yticks(arange(0.5,4.5),['sepal length',  'sepal width', 'petal length', 'petal width'],rotation=-20)
show()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pcad = pca.fit_transform(data)

plot(pcad[target=='setosa',0],pcad[target=='setosa',1],'bo')
plot(pcad[target=='versicolor',0],pcad[target=='versicolor',1],'ro')
plot(pcad[target=='virginica',0],pcad[target=='virginica',1],'go')
show()


