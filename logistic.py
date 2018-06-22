from sklearn.linear_model import LogisticRegression

#train and test data should be in matrix format. Pulled the target variable out of the train matrix

# Make the model with the specified regularization parameter
lr = LogisticRegression(C = 0.0001)

# Train on the training data
lr.fit(lr_train, target)
# Make predictions
# Make sure to select the second column only
lr_pred = lr.predict_proba(lr_test)[:, 1]
submit = test[['SK_ID_CURR']]
submit['TARGET'] = lr_pred
submit.to_csv("logistic regression.csv", index = False)