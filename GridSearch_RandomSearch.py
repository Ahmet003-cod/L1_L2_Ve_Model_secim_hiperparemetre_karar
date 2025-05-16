# -*- coding: utf-8 -*-
"""
Created on Tue May 13 22:48:26 2025

@author: Huzur Bilgisayar
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

knn=KNeighborsClassifier(n_neighbors=5)
knn_param_grid={"n_neighbors":np.arange(2,31)}
knn_grid_search=GridSearchCV(knn,knn_param_grid)
knn_grid_search.fit(X_train,y_train)
print("KNN Grid Search Best Parameters:",knn_grid_search.best_params_)
print("KNN Grid Search Best Accuary:",knn_grid_search.best_score_)


knn_grid_search=RandomizedSearchCV(knn,knn_param_grid,n_iter=29)
knn_grid_search.fit(X_train,y_train)
print("KNN Random Search Best Parameters:",knn_grid_search.best_params_)
print("KNN Random Search Best Accuary:",knn_grid_search.best_score_)
print()
#tree
d_tree=DecisionTreeClassifier()

d_tree_param_grid={"max_depth":[3,5,7],
                   "max_leaf_nodes":[None,5,10,20,30,50]}
d_tree_grid_search=GridSearchCV(d_tree,d_tree_param_grid)
d_tree_grid_search.fit(X_train,y_train)
print("DT Grid Search Best Parameters:",d_tree_grid_search.best_params_)
print("DT Grid Search Best Accuary:",d_tree_grid_search.best_score_)


d_tree_random_search=RandomizedSearchCV(d_tree,d_tree_param_grid)
d_tree_random_search.fit(X_train,y_train)
print("Dt Random Search Best Parameters:",d_tree_grid_search.best_params_)
print("DT Random Search Best Accuary:",d_tree_grid_search.best_score_)
print()
print()
#SVM
svm=SVC()
svm_param_grid={"C":[0.1,1,10,100],
                "gamma":[0.1,0.01,0.001,0.0001]}

svm_grid_search=GridSearchCV(svm,svm_param_grid)
svm_grid_search.fit(X_train,y_train)
print("SVM Grid Search Best Parameters:",svm_grid_search.best_params_)
print("SVM Grid Search Best Accuary:",svm_grid_search.best_score_)


svm_random_search=RandomizedSearchCV(svm,svm_param_grid)
svm_random_search.fit(X_train,y_train)
print("SVM Random Search Best Parameters:",svm_grid_search.best_params_)
print("SVM Random Search Best Accuary:",svm_grid_search.best_score_)























