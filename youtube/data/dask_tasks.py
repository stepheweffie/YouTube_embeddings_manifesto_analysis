import dask.dataframe as dd

df = dd.read_parquet("DataTask/*.parquet")
print(df.head(), df.columns)

data = df[df.cluster_label == 0]
print(data.head(), len(data))