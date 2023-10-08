#%%
import polars as pl
# %%
train = pl.read_csv("train.csv")
test = pl.read_csv("test.csv")
weather = pl.read_csv("weather.csv")
print(train.shape)
print(test.shape)
print(weather.shape)
# %%
train.head()
# %%
test.head()
# %%
weather.head()
# %%
merge = pl.concat([train, test], how="diagonal")
print(merge.shape)
merge
# %%
merge = (
    merge.with_columns(
        pl.col("date").cast(pl.Utf8).alias("date_str")
    ).with_columns(
        pl.col("date_str").str.slice(4, 2).alias("month"),
        pl.col("date_str").str.slice(6, 2).alias("day")
    ).with_columns(
        pl.col("month").cast(pl.Int8),
        pl.col("day").cast(pl.Int8)
    ).drop("date_str")
)
merge
# %%
merge = (
    merge.with_columns(
        pl.col("area").str.split(by="_")
    ).with_columns(
        [pl.col("area").list.get(i).alias(f"area{i+1}") for i in range(3)]
    ).drop("area")
)
merge
# %%
# merge = (
#     merge.join(
#         weather,
#         left_on=["date", "area1"],
#         right_on=["date", "area"],
#         how="left",
#         suffix="_area1"
#     ).join(
#         weather,
#         left_on=["date", "area2"],
#         right_on=["date", "area"],
#         how="left",
#         suffix="_area2"
#     ).join(
#         weather,
#         left_on=["date", "area3"],
#         right_on=["date", "area"],
#         how="left",
#         suffix="_area3"
#     ).select(
#             pl.exclude("^max_temp_time.*$", "^min_temp_time.*$")
#     )
# )
# merge
# %%
merge = merge.to_dummies(["area1", "area2", "area3"])
merge.head()
# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded = le.fit_transform(merge.get_column("kind"))
merge = (
    merge.with_columns(
        pl.lit(encoded).alias("kind")
    )
)
merge
# %%
# from sklearn.preprocessing import OrdinalEncoder
# oe = OrdinalEncoder()
# encoded = oe.fit_transform(merge.select("area1", "area2", "area3"))
# merge = (
#     merge.with_columns(
#         pl.lit(encoded[:, 0]).alias("area1"),
#         pl.lit(encoded[:, 1]).alias("area2"),
#         pl.lit(encoded[:, 2]).alias("area3")
#     )
# )
# merge
# %%
train = merge.head(69104)
test = merge.tail(220).drop("mode_price", "amount")
print(train.shape)
print(test.shape)
# %%
train.write_parquet("prepro_data/train.parquet")
test.write_parquet("prepro_data/test.parquet")
# %%
