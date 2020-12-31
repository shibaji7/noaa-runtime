import pandas as pd
from typing import Tuple

import numpy as np
import datetime as dt

from sklearn.linear_model import LogisticRegressionCV, LinearRegression
# import sys
# sys.path.append("libs/*")
# import skljson


import json
import pickle

import keras
import xgboost

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

def XGB(
    solar_wind_7d: pd.DataFrame,
    satellite_positions_7d: pd.DataFrame,
    latest_sunspot_number: float,
    ds: str,
) -> Tuple[float, float]:
    ### Change these parameters as needed
    ### Change these parameters as needed
    n_prin_comp = 25
    ndays_sw_input = 3
    col_list = ['bx_gsm', 'by_gsm', 'bz_gsm', 'density', 'speed', 'temperature', 'bt', 'dPhi_dt']
    
    # mean dst and sunspot values
    data_dir = "algorithms/XGB/" + ds + "/"
    dst_mean = -11.05
    dst_std = 19.07
    sun_spot_mean=60.
    ss_spot_std=53.
    # mean sw and imf
    sw_imf_mean_std_dict = {
        "bz_gsm": {"mean": -0.0291284300171361, "std": 3.430580041987825}, 
        "by_gsm": {"mean": 0.0884211011758524, "std": 3.9098851209499452}, 
        "bx_gsm": {"mean": -0.6608663272683737, "std": 3.667604711805854}, 
        "speed": {"mean": 430.58540311301726, "std": 100.5777282159303}, 
        "density": {"mean": 4.421725736076064, "std": 4.331910657238587}, 
        "temperature": {"mean": 115096.75503182354, "std": 120312.03957275226}, 
        "bt": {"mean": 1.1548875355396968, "std": 0.8162319614630422}, 
        "dPhi_dt": {"mean": 0.5526257446174042, "std": 1.1521844059182238}
    }
    ### Change these parameters as needed
    ### Change these parameters as needed
    
    # the original SW df may not have the timedelta index between 0 and 7 days!
    # so we'll set this index manually
    new_index = pd.timedelta_range(start='0 minutes', end='10079 minutes', freq='1min')
    solar_wind_7d.set_index(new_index, inplace=True)
    
    # create a timedelta range for the three days!
    end_minutes = str(ndays_sw_input * 24 * 60) + ' minutes'
#     solar_wind_7d.set_index()
    tdelta_range = pd.timedelta_range(start='0 minutes', end=end_minutes, freq='1min')
    # create a empty DF with this index
    empty_col = np.ones(tdelta_range.shape[0])
    empty_df = pd.DataFrame(data=empty_col,index=tdelta_range, columns=["empty_col"])
    
    solar_wind_7d = solar_wind_7d.interpolate(method='linear', axis=0).ffill().bfill()
    sel_df = solar_wind_7d.loc[tdelta_range.min():tdelta_range.max()]
    sel_df = sel_df.join(empty_df,how="outer")
#     sel_df.sort_index(inplace=True)
    # create bt and other vars
    sel_df["bt"] = np.sqrt(np.square(sel_df["by_gsm"]) + np.square(sel_df["bz_gsm"]))
    sel_df["theta_c"] = np.round(np.arctan2(sel_df["by_gsm"], sel_df["bz_gsm"]), 2) % (2*np.pi)
    sel_df["dPhi_dt"] = (sel_df["speed"]**(4./3)) * (sel_df["bt"] ** (2./3)) * (np.sin(sel_df["theta_c"] / 2.))**(8./3)
    
#     sel_df.interpolate(method='linear', limit_direction='both', inplace=True)
    sel_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sel_df = sel_df.interpolate(method='linear', axis=0).ffill().bfill()
    if sel_df.isnull().sum().sum() > sel_df.shape[0]:
        return (dst_mean,dst_mean)
    
    # Normalize the cols
    for _par in col_list:
        col_mean = sw_imf_mean_std_dict[_par]["mean"]
        col_std = sw_imf_mean_std_dict[_par]["std"]
        sel_df[_par] = (sel_df[_par] - col_mean) / col_std 
    norm_sunspot = (latest_sunspot_number - sun_spot_mean)/ss_spot_std
    sw_imf_values = sel_df[col_list].values.flatten().reshape(1,-1)
    # apply PCA to the sw imf values
    pca_reload = pickle.load(open(data_dir + "sw_imf_pca.pkl",'rb'))
# #     print("np array-->",sw_imf_values.shape)
    sw_pca = pca_reload.transform(sw_imf_values)
    norm_sunspot_arr = np.ones(sw_imf_values.shape[0]) * norm_sunspot
    input_data = np.append(sw_pca[:,:n_prin_comp],\
                              norm_sunspot_arr.reshape(\
                                  norm_sunspot_arr.shape[0],1),axis=1)
    # convert to Dmatrix for xgb
    input_data_dm = xgboost.DMatrix(input_data)
    # load the xgboost models
    bst_t0 = xgboost.Booster({'nthread': 4})  
    bst_t1 = xgboost.Booster({'nthread': 4})  
    bst_t0.load_model(data_dir + "dst_t0.model")
    bst_t1.load_model(data_dir + "dst_t1.model")
    
    prediction_at_t0 = bst_t0.predict(input_data_dm)
    prediction_at_t0 = prediction_at_t0*dst_std+dst_mean
    prediction_at_t1 = bst_t1.predict(input_data_dm)
    prediction_at_t1 = prediction_at_t1*dst_std+dst_mean
    print("prediction_at_t0,prediction_at_t1-->",prediction_at_t0[0], prediction_at_t1[0])
    
    return (prediction_at_t0[0], prediction_at_t1[0])

