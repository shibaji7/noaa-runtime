import pandas as pd
import datetime as dt
import numpy as np

sub = pd.read_csv("submission/submission.csv")
sub.timedelta = pd.to_timedelta(sub.timedelta)

dst = pd.read_csv("data/dst_labels.csv")
dst.timedelta = pd.to_timedelta(dst.timedelta)

print(" Length (before):", len(dst), len(sub))
new_df = sub.merge(dst,  how='inner', left_on=["period","timedelta"], right_on = ["period","timedelta"])
print(" Length (after):", len(new_df))
print(new_df.head())
print(" RMSE(t0) - ", (np.sqrt(np.mean((new_df.dst-new_df.t0)**2))))
dst_t1, t1 = np.roll(new_df.dst, -1)[:-1], np.array(new_df.t1)[:-1]
print(" RMSE(t1) - ", (np.sqrt(np.mean((dst_t1-t1)**2))))
