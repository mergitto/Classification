from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO

iris = load_iris()
#print(iris.data)
#print(iris.target)
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(iris.data, iris.target)
predicted = clf.predict(iris.data)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("./graph.pdf")

