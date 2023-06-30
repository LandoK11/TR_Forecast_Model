import pandas as pd
import numpy as np
import setup_env as env
from sklearn.utils import shuffle
import xgboost
import pickle
from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler
from sklearn import set_config
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime, date
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

results_df = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "AUC", "a", "b", "TrueNeg", "FalsePos", "FalseNeg", "TruePos" ])


df = pd.read_sql(env.str_sql_data, env.engine_sf_tandr)
df = df[(pd.to_datetime(df['Completed_Date']).dt.date > date(2022,1,1))] ######### DELETE THIS


df['Count'] = df.groupby('vin')['Completed_Date'].rank(ascending=True)
# scratch_df = df[df['vin'] == '1FVMG3DV9LHLY8392']
# test_df = df.sort_values('Count').groupby(['vin']).tail(1)

# df['registration_expiration_date'] = pd.to_datetime(df['registration_expiration_date'], errors='coerce')
df['Age_at_Expiry'] = pd.to_datetime(df['registration_expiration_date'], errors='coerce').dt.strftime('%Y')
df['registration_expiration_date'] = df['registration_expiration_date'].apply(lambda x : datetime.strptime(x, "%Y-%m-%d"))
# df = df[(df['registration_expiration_date'] < pd.Timestamp.today())]

def IfExpired(row):
    if row['Completed_Date'] > row['registration_expiration_date']:
        return 'Expired'
    else:
        return 'Valid'

df['Expiry_Flag'] = df.apply(IfExpired, axis=1)

def RenewAgain(row):
    if pd.isna(row['Next_Complete']):
        return 0
    else:
        return 1

df['IF_RENEW_AGAIN'] = df.apply(RenewAgain, axis=1)

input_data = df[['Completed_Date', 'registration_expiration_date', 'Work_Requested', 'VehicleYear', 'Customer', 'CountyId', 'State', 'Count', 'Age_at_Expiry', 'IF_RENEW_AGAIN']]
# input_data = df[['registration_expiration_date', 'VehicleYear', 'Make', 'Customer', 'CountyId', 'Count', 'IF_RENEW_AGAIN']]
# input_data = df[['registration_expiration_date', 'VehicleYear', 'Make', 'Customer', 'CountyId', 'IF_RENEW_AGAIN']]

def SeenBefore(row):
    if row['Count'] > 1:
        return 'Yes'
    else:
        return 'No'

input_data['IF_SEEN_BEFORE'] = df.apply(SeenBefore, axis=1)


input_data['month'] = input_data.apply(lambda x: str(x['registration_expiration_date'].month), axis=1)

input_data['CountyId'] = input_data['CountyId'].astype(str)
input_data['month'] = input_data['month'].astype(int)


input_data['VehicleYear'] = pd.to_numeric(input_data['VehicleYear'], errors='coerce')
input_data['Customer'].replace('', np.nan, inplace=True)
# input_data = input_data[(input_data['Count'] < 20)]
input_data.dropna(inplace=True)

input_data[['Age_at_Expiry', 'VehicleYear']] = input_data[['Age_at_Expiry', 'VehicleYear']].astype(int)
input_data['Age_at_Expiry'] = input_data['Age_at_Expiry'] - input_data['VehicleYear']

hold_out = input_data[pd.to_datetime(input_data['registration_expiration_date']).dt.date >= date(2023,5,1)]
input_data = input_data[pd.to_datetime(input_data['registration_expiration_date']).dt.date < date(2023,3,1)]
# shuffle
input_data = shuffle(input_data)


# select features and target for model
# input_data = input_data[['registration_expiration_date', 'VehicleYear', 'Make', 'Customer', 'CountyId', 'Count', 'IF_RENEW_AGAIN']]
input_data = input_data.drop(columns=['Completed_Date','registration_expiration_date', 'VehicleYear'])

# input_data['registration_expiration_date'] = pd.to_datetime(input_data['registration_expiration_date']).dt.date
#
# train_set = input_data[input_data.registration_expiration_date < date(2023,1,1)]
# test_set = input_data[input_data.registration_expiration_date >= date(2023,1,1)]
#
# train_set = train_set.drop(columns=['registration_expiration_date', 'VehicleYear'])
# test_set = test_set.drop(columns=['registration_expiration_date', 'VehicleYear'])
train_set, test_set = np.split(input_data, [int(.70 * len(input_data))])

X_train = pd.DataFrame(train_set.drop(['IF_RENEW_AGAIN'], axis=1))
y_train = train_set['IF_RENEW_AGAIN']

X_test = pd.DataFrame(test_set.drop(['IF_RENEW_AGAIN'], axis=1))
y_test = test_set['IF_RENEW_AGAIN']



##### PIPELINE ######
cat_columns = ['Work_Requested', 'Customer', 'CountyId', 'State', 'IF_SEEN_BEFORE']

num_columns = ['Age_at_Expiry', 'Count', 'month']

scaler = Pipeline(steps=[("scaler", StandardScaler())])

#set up transformer for OHE and scaling
ohe = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", scaler, num_columns),
        ("cat", ohe, cat_columns),
    ],
             remainder='passthrough', verbose_feature_names_out=False,
)


model = xgboost.XGBClassifier(objective='binary:logistic', booster='gbtree', eval_metric='auc',
                                tree_method='hist', grow_policy='lossguide', use_label_encoder=False, gamma=.1,
                                learning_rate=.3, max_depth=14, n_estimators=115, reg_alpha=.8, reg_lambda=.1)

pipe = Pipeline(
    steps=[("prep", preprocess), ("model", model)])

# Display Pipeline
set_config(display='diagram')

print("Training model...")
# fit pipeline to train data (model fitting)
pipe.fit(X_train, y_train)

print("Model fit successfully")

print("Performing CV...")
# 5-fold CV of fitted pipeline model
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
scores = cross_val_score(pipe, X_train, y_train, scoring='r2', cv=cv)
mean_cv = scores.mean()
print("CV Complete")

# make predictions
preds = pipe.predict(X_test)
test_score = pipe.score(X_test, y_test)
probs = pipe.predict_proba(X_test)[:, 1]

test_df = pd.DataFrame({'Prob':probs, 'Prediction':preds, 'Actual':y_test})

results_df = pd.concat([X_test,test_df], axis=1)







fig, axes = plt.subplots(3, figsize=(10,20))

# calibration diagram
fop, mpv = calibration_curve(y_test, probs, n_bins=10, normalize=True)
# plot perfectly calibrated
axes[0].plot([0, 1], [0, 1], linestyle='--')
# plot calibrated reliability
axes[0].plot(mpv, fop, marker='.')
axes[0].set_title('Calibration Curve')
axes[0].set_xlabel('Mean predicted prob')
axes[0].set_ylabel('Fraction of positives')

# Histo's

axes[1] = test_df.pivot(columns='Actual').Prob.plot(kind= 'hist', stacked=True, ax=axes[1])
for c in axes[1].containers:
    axes[1].bar_label(c, label_type='center')
axes[1].set_title('Histogram')
axes[1].set_xlabel('Mean predicted prob')
axes[1].set_ylabel('Count')

# CONFUSION MATRIX PLOT
cnf_matrix = metrics.confusion_matrix(y_test, preds)
class_names = [0, 1]  # name  of classes
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', ax=axes[2])
# axes[2, 0].xaxis.set_label_position("top")
axes[2].set_title('Confusion Matrix')
axes[2].set_ylabel('Actual label')
axes[2].set_xlabel('Predicted label')

fig.tight_layout()
fig.show()


# #### Permutation Importance ####
# result = permutation_importance(
#     pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
# )
#
# X = X_test[cat_columns + num_columns]
#
# sorted_importances_idx = result.importances_mean.argsort()
# importances = pd.DataFrame(
#     result.importances[sorted_importances_idx].T,
#     columns=X.columns[sorted_importances_idx],
# )
# ax = importances.plot.box(vert=False, whis=10)
# ax.set_title("Permutation Importances (test set)")
# ax.axvline(x=0, color="k", linestyle="--")
# ax.set_xlabel("Decrease in accuracy score")
# ax.figure.tight_layout()
# plt.show()
#




# AOC Curve
fpr, tpr, _ = metrics.roc_curve(y_test, probs)
auc = metrics.roc_auc_score(y_test, probs)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.title("New Model AUC")
plt.legend(loc=4)
plt.show()



analysis_df = results_df[results_df.Prob > .70]

hold_out = hold_out[(hold_out['registration_expiration_date'] < pd.Timestamp.today())]
hold_out_input = hold_out.drop(columns=['Completed_Date','registration_expiration_date', 'VehicleYear', 'IF_RENEW_AGAIN'])
# make predictions
preds = pipe.predict(hold_out_input)
probs = pipe.predict_proba(hold_out_input)[:, 1]

test_df = pd.DataFrame({'Prob':probs, 'Prediction':preds})
hold_out = hold_out.reset_index()
results_df_2 = pd.concat([hold_out,test_df], axis=1)
results_df_2['registration_expiration_date'] = pd.to_datetime(results_df_2['registration_expiration_date'])
results_df_2['month_year'] = results_df_2['registration_expiration_date'].dt.to_period('M')

results_df_2 = results_df_2[results_df_2['Prediction'] == 1]
print(results_df_2['month_year'].value_counts())