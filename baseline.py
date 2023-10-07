#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgbm
import japanize_matplotlib
mpl.style.use("ggplot")
import datetime as dt
#%%
# データの読み込み
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train = train.drop(columns="amount")
print(train.info())
print(train.shape)
print(test.shape)
train.head()
# %%
# 学習データとテストデータを結合してから，質的変数をOne-Hotベクトル化
merge = pd.concat([train, test])
merge = pd.get_dummies(merge)

# s_format = '%Y%m%d'
# merge["date"] = merge["date"].apply(lambda x: str(x))
# merge["date"] = merge["date"].apply(lambda x: dt.datetime.strptime(x, s_format))

train = merge.iloc[:69104, :]
test = merge.iloc[69104:, :]
# %%
# 検証用データの準備
train, valid = train_test_split(train, test_size=0.25, random_state=97)

y_train = train["mode_price"]
X_train = train.drop(columns="mode_price")

y_valid = valid["mode_price"]
X_valid = valid.drop(columns="mode_price")
#%%
# LightGBM
train_set = lgbm.Dataset(X_train, y_train)
valid_set = lgbm.Dataset(X_valid, y_valid)

params = {"objective": "regression",
          "metric":"rmse",
          "learning_rate":0.05,
          "num_iterations":1000
          }
# history = {}
model = lgbm.train(
    params = params,
    train_set = train_set,
    valid_sets = [train_set, valid_set])
# %%
# 予測
pred = model.predict(X_valid)
print("RMSPE:", np.sqrt(np.mean(((pred - y_valid) / y_valid)**2))*100)
# print("r2:", r2_score(y_valid, pred))
plt.scatter(y_valid, pred)
# %%
# plt.plot(history["training"]["rmse"], label = "train")
# plt.plot(history["valid_1"]["rmse"], label = "valid")
# plt.title("学習曲線")
# plt.legend()
# %%
# テストデータで予測
test = test.drop(columns="mode_price")
pred = model.predict(test)
# %%
# 提出データの作成
submit = pd.read_csv("sample_submission.csv")
submit["mode_price"] = pred
submit.to_csv("submission.csv", index=False)
# %%
test.shape
# %%
