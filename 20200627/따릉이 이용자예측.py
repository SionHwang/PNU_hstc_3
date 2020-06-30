import pandas as pd #판다스 패키지 포함
from sklearn.ensemble import RandomForestRegressor #랜덤포레스트 포함

'''----파일호출----'''
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('submission.csv')

'''---전처리---'''
train['hour_bef_temperature'].fillna({934:14.783333, 1035:20.895082},inplace = True)
train['hour_bef_windspeed'].fillna({18:3.281356, 244:1.836667, 260:1.9655517, 376:1.965517, 780:3.278333, 934:1.965517, 1035:3.838333, 1138:2.766667, 1229:1.633333},inplace = True)
train['hour_bef_precipitation'].fillna({934: 0.016949, 1035:0.016667},inplace = True)
train['hour_bef_visibility'].fillna({934:1434.220339, 1035:1581.850000},inplace = True)
train['hour_bef_humidity'].fillna({934:58.169492, 1035:40.450000},inplace = True)

test['hour_bef_temperature'].fillna(19.704918, inplace = True)
test['hour_bef_windspeed'].fillna(3.595072, inplace = True)
test['hour_bef_precipitation'].fillna(0.068966, inplace = True)
test['hour_bef_visibility'].fillna(1561.758621, inplace = True)
test['hour_bef_humidity'].fillna(47.689655, inplace = True)

train[train['hour_bef_temperature'].isna()].index
train[train['hour_bef_windspeed'].isna()].index
train[train['hour_bef_precipitation'].isna()].index
train[train['hour_bef_visibility'].isna()].index
train[train['hour_bef_humidity'].isna()].index

features = ['hour','hour_bef_temperature','hour_bef_precipitation','hour_bef_windspeed','hour_bef_visibility','hour_bef_humidity']
X_train = train[features]
y_train = train['count']
X_test = test[features]


model= RandomForestRegressor(n_estimators=1450, max_depth = 725, random_state=0, n_jobs = -1)


model.fit(X_train, y_train)
ypred2 = model.predict(X_test)

'''----파일저장----'''
submission['count'] = ypred2
submission.to_csv('model_output.csv', index=False)
