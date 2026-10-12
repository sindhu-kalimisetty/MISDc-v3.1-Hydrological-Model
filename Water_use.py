import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_water_use(
    time_period,
    sector='both',
    base_demand_civil=0.133,
    base_demand_industrial=0.089,
    seasonal_factor=True,
    # daily_pattern=True,
    efficiency_factor=0.8,
    climate_zone='temperate',
    industrial_type='mixed'
    ):

    time_period = np.array(time_period)
    if time_period.ndim == 2:
        M, N = time_period.shape
    else:
        M = len(time_period)
        N = 1
        time_period = time_period.reshape(M, N)

    water_use = np.zeros((M, N))
    if np.issubdtype(time_period.dtype, np.datetime64):
        months = pd.DatetimeIndex(time_period.flatten()).month.to_numpy()
    elif isinstance(time_period[0, 0] if time_period.ndim == 2 else time_period[0], (pd.Timestamp, )):
        months = pd.DatetimeIndex(time_period.flatten()).month.to_numpy()
    else:
        raise TypeError("time_period must be a datetime-like array")

    months = months.reshape(M, N)

    # Define seasonal factors
    seasonal_factors_civil = {
        'temperate': {
            1: 0.11, 2: 0.10, 3: 0.11, 4: 0.26, 5: 0.46, 6: 1.0,
            7: 0.98, 8: 0.52, 9: 0.35, 10: 0.13, 11: 0.12, 12: 0.12
        }     
    }
    
    seasonal_factors_industrial = {
        'mixed': {
            1: 0.11, 2: 0.10, 3: 0.11, 4: 0.26, 5: 0.46, 6: 1.0,
            7: 0.98, 8: 0.52, 9: 0.35, 10: 0.13, 11: 0.12, 12: 0.12
        }
    }

    # Calculate demands for each point
    for i in range(M):
        for j in range(N):
            month = months[i, j]
            civil_demand = 0
            industrial_demand = 0

            # Calculate civil demand
            if sector in ['civil', 'both']:
                civil_demand = base_demand_civil
                if seasonal_factor:
                    civil_demand *= seasonal_factors_civil[climate_zone][month]
                civil_demand /= efficiency_factor

            # Calculate industrial demand
            if sector in ['industrial', 'both']:
                industrial_demand = base_demand_industrial
                if seasonal_factor:
                    industrial_demand *= seasonal_factors_industrial[industrial_type][month]
                industrial_demand /= efficiency_factor

            # Total demand
            water_use[i, j] = civil_demand + industrial_demand

    # Add random variations (±10%)
    np.random.seed(42)  
    random_factors = np.random.uniform(0.9, 1.1, size=(M, N))
    water_use *= random_factors
    return water_use




