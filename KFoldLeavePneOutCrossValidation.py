# -*- coding: utf-8 -*-
"""
Created on Wed May 14 19:43:27 2025
@author: Huzur Bilgisayar
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,KFold,LeaveOneOut,GridSearchCV
from sklearn.tree import DecisionTreeClassifier


iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

tree=DecisionTreeClassifier()
tree_param_dist={"max_depth":[3,5,7]}

#KFOLD GRİD SEARCH
kf=KFold(n_splits=10)
tree_grid_search=GridSearchCV(tree, tree_param_dist,cv=kf)
tree_grid_search.fit(X_train,y_train)
print("KFOLD En iyi parametre=:",tree_grid_search.best_params_)
print("KFOLD   En iyi accuary=:",tree_grid_search.best_score_)
print("\n")
#LeavePOut
loo=LeaveOneOut()
tree_grid_search_loo=GridSearchCV(tree, tree_param_dist,cv=loo)
tree_grid_search_loo.fit(X_train,y_train)

print("LeavePOut En iyi parametre=:",tree_grid_search_loo.best_params_)
print("LeavePOut   En iyi accuary=:",tree_grid_search_loo.best_score_)