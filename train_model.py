# %%
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgbm
import japanize_matplotlib
mpl.style.use("ggplot")
# %%
train = pl.read_parquet("prepro_data/train.parquet")
test = pl.read_parquet("prepro_data/test.parquet")
print(train.shape)
print(test.shape)
# %%
train_X = train.drop("amount", "mode_price").to_pandas()
train_y = train.select("mode_price").to_pandas()
test = test.to_pandas()
# %%
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import tqdm

models = []
scores_MSE = []

va_y_list = np.array([])
y_pred_list = np.array([])

kf = KFold(n_splits=5, shuffle=True, random_state=97)
for tr_idx, va_idx in kf.split(train_X):
    tr_x, va_x = train_X.iloc[tr_idx], train_X.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_valid = lgb.Dataset(va_x, va_y, reference=lgb_train)

    params = {"objective" : "regression",
              'metric' : 'rmse',
              'early_stopping_rounds' : 10,
              "learning_rate" : 0.01,
              'n_estimators' : 5000}

    model = lgb.train(params=params,
                      train_set=lgb_train,
                      valid_sets=[lgb_train, lgb_valid])

    models.append(model)
    y_pred = model.predict(va_x, num_iteration=model.best_iteration)

    va_y_list = np.append(va_y_list, va_y)
    y_pred_list = np.append(y_pred_list, y_pred)

    mse = mean_squared_error(va_y, y_pred)

    scores_MSE.append(mse)

np.sqrt(scores_MSE)
# %%
print("RMSPE:", np.sqrt(np.mean(((y_pred_list - va_y_list) / va_y_list)**2))*100)
# %%
# テストデータで予測
pred = model.predict(test)
# %%
# 提出データの作成
submit = pd.read_csv("sample_submission.csv")
submit["mode_price"] = pred
submit.to_csv("submission.csv", index=False)
# %%
