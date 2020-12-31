import os
import sys
import pandas as pd
import datetime as dt
import numpy as np

import argparse
import dateutil
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algo", required=True, help="Algorithm names [LR, LSTM, XGB, GB]")
    parser.add_argument("-t", "--ds", required=True, help="Dataset name [2h, 6h, 24h, 72h]")
    parser.add_argument("-d", "--desc", default="Algorithm description", help="Description in terms of training")
    args = parser.parse_args()

    os.system("mkdir -p RMSE_estimates_local/{algo}/{ds}/".format(algo=args.algo, ds=args.ds))
    os.system("cp submission/submission.csv RMSE_estimates_local/{algo}.{ds}.submission.csv".format(algo=args.algo, ds=args.ds))
    
    sub = pd.read_csv("submission/submission.csv")
    sub.timedelta = pd.to_timedelta(sub.timedelta)
    
    dst = pd.read_csv("data/dst_labels.csv")
    dst.timedelta = pd.to_timedelta(dst.timedelta)
    
    print(" Length (before):", len(dst), len(sub))
    new_df = sub.merge(dst,  how='inner', left_on=["period","timedelta"], right_on = ["period","timedelta"])
    print(" Length (after):", len(new_df))
    print(" RMSE(t0) - ", (np.sqrt(np.mean((new_df.dst-new_df.t0)**2))))
    dst_t1, t1 = np.roll(new_df.dst, -1)[:-1], np.array(new_df.t1)[:-1]
    print(" RMSE(t1) - ", (np.sqrt(np.mean((dst_t1-t1)**2))))
    
    rdme = {
        "desc": args.desc,
        "ds": args.ds,
        "algo": args.algo,
        "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "RMSE(t0)": "%.4f"%(np.sqrt(np.mean((new_df.dst-new_df.t0)**2))),
        "RMSE(t1)": "%.4f"%(np.sqrt(np.mean((dst_t1-t1)**2)))
    }
    with open("RMSE_estimates_local/{algo}.{ds}.README.json".format(algo=args.algo, ds=args.ds), "w") as f:
        o = json.dumps(rdme, sort_keys=True, indent=4)
        f.write(o)