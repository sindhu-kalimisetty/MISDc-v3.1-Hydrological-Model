import numpy as np
import pandas as pd
from klingupta import klinggupta

def perf(Qsim,Qobs):
    RMSE = np.sqrt(np.nanmean((Qsim - Qobs)**2))
    NS = 1-(np.nansum((Qsim - Qobs)**2)/np.nansum((Qobs - np.nanmean(Qobs))**2))
    ANSE = 1 - (np.nansum((Qobs + np.nanmean(Qobs)) * (Qsim - Qobs)**2) /
                np.nansum((Qobs + np.nanmean(Qobs)) * (Qobs - np.nanmean(Qobs))**2))
    NS_radQ = 1 - (np.nansum((np.sqrt(Qsim) - np.sqrt(Qobs))**2) /
                    np.nansum((np.sqrt(Qobs) - np.nanmean(np.sqrt(Qobs)))**2))
    NS_lnQ = 1 - (np.nansum((np.log(Qsim + 0.00001) - np.log(Qobs + 0.00001))**2) /
                    np.nansum((np.log(Qobs + 0.00001) - 
                            np.nanmean(np.log(Qobs + 0.00001)))**2))

    def matlab_style_corrcoef(x, y):
        x = np.array(x)
        y = np.array(y)
        mask = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[mask]
        y_valid = y[mask]
        if len(x_valid) > 0:
            return np.corrcoef(x_valid, y_valid)[0, 1]
        else:
            return np.nan
    RQ = matlab_style_corrcoef(Qobs, Qsim)
    KGE_out = klinggupta(Qsim, Qobs)
    return NS, RMSE, ANSE, RQ, NS_lnQ, NS_radQ, KGE_out