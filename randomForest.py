# train variable should be a matrix
# This is a pretty simple model to fit
# I don't like this model very much. I would like to try h2o's random forest and see how it predicts


from sklearn.ensemble import RandomForestClassifier
rf_train,rf_test = impute(train,test)
rf = RandomForestClassifier(n_estimators = 100, random_state = 50,criterion='entropy', oob_score=True)
rf.fit(rf_train, target)
pred = rf.predict_proba(rf_test)[:,1]
entry = test[['SK_ID_CURR']]
entry['TARGET'] = pred
entry.to_csv('randomForest.csv', index = False)