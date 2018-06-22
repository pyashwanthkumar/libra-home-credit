import lightgbm as lgbm

# All the categorical features are label encoded. We don't need to one-hot encode for the light gbm as it partions each category into two subset. This solution has time complexity of the order O(n*log(n)). n - number of categories in each feature. It's lower than that of one-hot encoding.

# Using .Dataset() function to load the training data faster.

# fin_train is the training dataframe without target. 

# fin_target is our target variable. Code for extracting it is below this line.

#if 'TARGET' in fin_train:
#   fin_target = fin_train.pop('TARGET')

# I decided learning rate 0.1 would be decent enough considering the amount of time it takes to train this data.

cat_features_fin = fin_train.columns[fin_train.dtypes == 'object']
lgbm_train = lgbm.Dataset(data = fin_train,label=fin_target, categorical_feature=cat_features_fin.tolist(), free_raw_data=False)
lgbm_params = {
    'boosting': 'dart',
    'application': 'binary',
    'learning_rate': 0.1,
    'min_data_in_leaf': 30,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.5,
    'scale_pos_weight': 2,
    'drop_rate': 0.02
}

cv_results = lgbm.cv(train_set=lgbm_train,params=lgbm_params,nfold=5, num_boost_round=600, early_stopping_rounds=50,verbose_eval=50,metrics=['auc'])


optimum_boost_rounds = np.argmax(cv_results['auc-mean'])
print('Optimum boost rounds = {}'.format(optimum_boost_rounds))
print('Best CV result = {}'.format(np.max(cv_results['auc-mean'])))

clf = lgbm.train(train_set=lgbm_train,
                 params=lgbm_params,
                 num_boost_round=optimum_boost_rounds)

""" Predict on test set and create submission """
y_pred = clf.predict(fin_test)
out_df = pd.DataFrame({'SK_ID_CURR': test['SK_ID_CURR'], 'TARGET': y_pred})
out_df.to_csv('submission_lgbm.csv', index=False)

fig, (ax, ax1) = plt.subplots(1, 2, figsize=[11, 7])
lgbm.plot_importance(clf, ax=ax, max_num_features=20, importance_type='split')
lgbm.plot_importance(clf, ax=ax1, max_num_features=20, importance_type='gain')
ax.set_title('Importance by splits')
ax1.set_title('Importance by gain')
plt.tight_layout()
plt.savefig('feature_importance.png')