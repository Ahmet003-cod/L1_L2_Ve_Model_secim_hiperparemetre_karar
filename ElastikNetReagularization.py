# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:19:30 2025

@author: Huzur Bilgisayar
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model  import ElasticNet
from sklearn.metrics import mean_squared_error

diabets=load_diabetes()
X=diabets.data
y=diabets.target

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

elastic_net=ElasticNet()
elastic_net_param_grid={"alpha":[0.1,1,10,100],
                        "l1_ratio":[0.1,0.3,0.5,0.7,0.9]}#{l1 or l2 penalty}
elastik_net_grid_search=GridSearchCV(elastic_net,elastic_net_param_grid,cv=5)
elastik_net_grid_search.fit(X_train,y_train)

print("ElasticNet en iyi sonuc:",elastik_net_grid_search.best_params_)
print("ElasticNet en iyi score:",elastik_net_grid_search.best_score_)

best_elastic_model=elastik_net_grid_search.best_estimator_
y_pred_elastik_net=best_elastic_model.predict(X_test)

elastic_net_mse=mean_squared_error(y_test, y_pred_elastik_net)
print()
print("elastic_net_mse:",elastic_net_mse)

