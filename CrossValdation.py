# -*- coding: utf-8 -*-
"""
Created on Wed May 14 18:11:11 2025
@author: Huzur Bilgisayar
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

#DT
nb_cv=3
tree=DecisionTreeClassifier()
tree_pram_grid={"max_depth":[3,5,7],
                "max_leaf_nodes":[None,5,10,20,30,50]}
tree_grid_search=GridSearchCV(tree,tree_pram_grid ,cv=nb_cv)
tree_grid_search.fit(X_train,y_train)
print("DT Grid Search Best Parameters:",tree_grid_search.best_params_)
print("DT Grid Search Best Accuary:",tree_grid_search.best_score_)

for mean_scor,params in zip(tree_grid_search.cv_results_["mean_test_score"],tree_grid_search.cv_results_["params"]):
    print(f"Ortalama test scoru:{mean_scor}, Parametreler:{params}")
print()
print()
cv_results=tree_grid_search.cv_results_
for i,params in enumerate((cv_results["params"])):
    print(f"parametreler={params}")
    for j in range(nb_cv):
        accuary=cv_results[f"split{j}_test_score"][i]
        print(f"\tFold {j+1}--Accuary={accuary}")
    

    