import pandas as pd
from typing import Tuple

import numpy as np
import datetime as dt

from models import LSTM, LR, XGB

REG_NAME = "LR" # [LR, LSTM, GB, XGB]
DATA_SET = "72h" # [0.5h, 1h, 2h, 6h, 24h, 72h]

# THIS MUST BE DEFINED FOR YOUR SUBMISSION TO RUN
def predict_dst(
    solar_wind_7d: pd.DataFrame,
    satellite_positions_7d: pd.DataFrame,
    latest_sunspot_number: float,
) -> Tuple[float, float]:
    """
    Take all of the data up until time t-1, and then make predictions for
    times t and t+1.
    Parameters
    ----------
    solar_wind_7d: pd.DataFrame
        The last 7 days of satellite data up until (t - 1) minutes [exclusive of t]
    satellite_positions_7d: pd.DataFrame
        The last 7 days of satellite position data up until the present time [inclusive of t]
    latest_sunspot_number: float
        The latest monthly sunspot number (SSN) to be available
    Returns
    -------
    predictions : Tuple[float, float]
        A tuple of two predictions, for (t and t + 1 hour) respectively; these should
        be between -2,000 and 500.
    """
    
    # Make a prediction
    if REG_NAME == "LSTM": prediction_at_t0, prediction_at_t1 = LSTM(solar_wind_7d, satellite_positions_7d, latest_sunspot_number, DATA_SET)
    if REG_NAME == "LR": prediction_at_t0, prediction_at_t1 = LR(solar_wind_7d, satellite_positions_7d, latest_sunspot_number, DATA_SET)
    if REG_NAME == "XGB": prediction_at_t0, prediction_at_t1 = XGB(solar_wind_7d, satellite_positions_7d, latest_sunspot_number, DATA_SET)
    
    return prediction_at_t0, prediction_at_t1
