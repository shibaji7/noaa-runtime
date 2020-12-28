import pandas as pd
from typing import Tuple

import numpy as np
import datetime as dt

from sklearn.linear_model import LogisticRegressionCV, LinearRegression
import sys
sys.path.append("libs/*")
import skljson


import json
import pickle

import keras

def check_dst_lims( 
    dst: float,
    lims: list = [500, -2000]
) -> float:
    """ 
    Limits outputs to [500, -2000] nT range.
    """
    if dst > lims[0]: dst = lims[0]
    if dst < lims[1]: dst = lims[1]
    return dst

# Define functions for preprocessing
def impute_features(feature_df):
    """Imputes data using the following methods:
    - `smoothed_ssn`: forward fill
    - `solar_wind`: interpolation
    """
    # forward fill sunspot data for the rest of the month
    feature_df.smoothed_ssn = feature_df.smoothed_ssn.fillna(method="ffill")
    # interpolate between missing solar wind values
    feature_df = feature_df.interpolate()
    return feature_df


def aggregate_hourly(feature_df, aggs=["mean", "std"]):
    """Aggregates features to the floor of each hour using mean and standard deviation.
    e.g. All values from "11:00:00" to "11:59:00" will be aggregated to "11:00:00".
    """
    # group by the floor of each hour use timedelta index
    agged = feature_df.groupby(
        [feature_df.index.get_level_values(0).floor("H")]
    ).agg(aggs)
    # flatten hierachical column index
    agged.columns = ["_".join(x) for x in agged.columns]
    return agged


def preprocess_features(solar_wind, sunspots, scaler=None, subset=None):
    """
    Preprocessing steps:
        - Subset the data
        - Aggregate hourly
        - Join solar wind and sunspot data
        - Scale using standard scaler
        - Impute missing values
    """
    # select features we want to use
    if subset:
        solar_wind = solar_wind[subset]

    # aggregate solar wind data and join with sunspots
    hourly_features = aggregate_hourly(solar_wind).join(sunspots)

    # subtract mean and divide by standard deviation
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(hourly_features)

    normalized = pd.DataFrame(
        scaler.transform(hourly_features),
        index=hourly_features.index,
        columns=hourly_features.columns,
    )

    # impute missing values
    imputed = impute_features(normalized)

    # we want to return the scaler object as well to use later during prediction
    return imputed, scaler


def LSTM(
    solar_wind_7d: pd.DataFrame,
    satellite_positions_7d: pd.DataFrame,
    latest_sunspot_number: float,
    ds: str,
) -> Tuple[float, float]:
    # Load in serialized model, config, and scaler
    model = keras.models.load_model("algorithms/LSTM/{ds}/model".format(ds=ds))
    
    with open("algorithms/LSTM/{ds}/config.json".format(ds=ds), "r") as f:
        CONFIG = json.load(f)
        
    with open("algorithms/LSTM/{ds}/scaler.pck".format(ds=ds), "rb") as f:
        scaler = pickle.load(f)

    # Set global variables    
    TIMESTEPS = CONFIG["timesteps"]
    SOLAR_WIND_FEATURES = [
        "bt",
        "temperature",
        "bx_gse",
        "by_gse",
        "bz_gse",
        "speed",
        "density",
    ]
    XCOLS = (
        [col + "_mean" for col in SOLAR_WIND_FEATURES]
        + [col + "_std" for col in SOLAR_WIND_FEATURES]
        + ["smoothed_ssn"]
    )

    # Re-format data to fit into our pipeline
    sunspots = pd.DataFrame(index=solar_wind_7d.index, columns=["smoothed_ssn"])
    sunspots["smoothed_ssn"] = latest_sunspot_number
    
    # Process our features and grab last 32 (timesteps) hours
    features, s = preprocess_features(
        solar_wind_7d, sunspots, scaler=scaler, subset=SOLAR_WIND_FEATURES
    )
    model_input = features[-TIMESTEPS:][XCOLS].values.reshape(
        (1, TIMESTEPS, features.shape[1])
    )
    # Make a prediction
    prediction_at_t0, prediction_at_t1 = model.predict(model_input)[0]
    print(" Test - ", prediction_at_t0, prediction_at_t1)
    
    # Optional check for unexpected values
    if not np.isfinite(prediction_at_t0): prediction_at_t0 = -15.
    if not np.isfinite(prediction_at_t1): prediction_at_t1 = -15.

    prediction_at_t0 = check_dst_lims(prediction_at_t0)
    prediction_at_t1 = check_dst_lims(prediction_at_t1)
    
    return prediction_at_t0, prediction_at_t1


def load_model(fname, ds):
    model = skljson.from_json("algorithms/LSTM/{ds}/".format(ds=ds) + fname)
    return model

def pandas_ts_nan_helper(
    dat: pd.DataFrame(),
    params: list
) -> pd.DataFrame():
    """
    Replace nans and 1H resample
    """
    for p in params:
        u = np.array(dat[p])
        xb, s = np.nanmean(u), np.nanstd(u)
        u[np.isnan(u)] = np.random.normal(xb, s, size=np.count_nonzero(np.isnan(u).astype(int)))
        dat[p] = u
        
    dat["time"] = [dt.datetime(2000,1,1) + dt.timedelta(minutes=i) for i in range(len(dat))]
    dat = dat.set_index("time")
    dat = ((dat.copy())[params]).dropna()
    dat = dat.resample("3600s").interpolate().reset_index()
    return dat.iloc[0]

def LR(
    solar_wind_7d: pd.DataFrame,
    satellite_positions_7d: pd.DataFrame,
    latest_sunspot_number: float,
    ds: str,
) -> Tuple[float, float]:
    params = ["bx_gse", "by_gse", "bz_gse", "theta_gse","phi_gse", "bx_gsm", "by_gsm", 
              "bz_gsm", "theta_gsm", "phi_gsm","bt", "density", "speed", "temperature"]
    sw = pandas_ts_nan_helper(solar_wind_7d, params)
    Xs = sw[params].values
    Xs = Xs.reshape((1, len(params)))
    print(" Shape: ", Xs.shape)
    dst = []
    
    for t in range(2):
        print(" T(%s) instance ... "%t)
        clf = load_model("clf_t%d.json"%t)
        reg_0, reg_1 = load_model("reg_t%d_0.json"%t), load_model("reg_t%d_1.json"%t)
        tag = ((clf.predict_proba(Xs) < 0.5).astype(int))[0,0]
        print( "Prob. (No-Storm):%.2f"%clf.predict_proba(Xs)[0,0], " - Tag: ", tag )
        if tag == 0: dst.append(reg_0.predict(Xs)[0])
        if tag == 1: dst.append(reg_1.predict(Xs)[0])
    print(" Predictions Dst - ",tuple(dst))
    return tuple(dst)

