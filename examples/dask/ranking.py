import dask.array as da
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_svmlight_file
from distributed import Client, LocalCluster

if __name__ == "__main__":

    print("loading data")

    X, y = load_svmlight_file("../lambdarank/rank.train")
    group = np.loadtxt("../lambdarank/rank.train.query")

    print("initializing a Dask cluster")

    cluster = LocalCluster(n_workers=2)
    client = Client(cluster)

    print("created a Dask LocalCluster")

    print("distributing training data on the Dask cluster")

    # split training data into two partitions
    rows_in_part1 = int(np.sum(group[:100]))
    rows_in_part2 = X.shape[0] - rows_in_part1
    num_features = X.shape[1]

    # make this array dense because we're splitting across
    # a sparse boundary to partition the data
    X = X.todense()

    dX = da.from_array(
        x=X,
        chunks=[
            (rows_in_part1, rows_in_part2),
            (num_features, )
        ]
    )
    dy = da.from_array(
        x=y,
        chunks=[
            (rows_in_part1, rows_in_part2),
        ]
    )
    dg = da.from_array(
        x=group,
        chunks=[
            (100, group.size - 100)
        ]
    )

    print("beginning training")

    dask_model = lgb.DaskLGBMRanker()
    dask_model.fit(dX, dy, group=dg)

    print("done training")