import numpy as np
from scipy.optimize import minimize
from MISD_SEMIDISTR_continuous_model import MISD_SEMIDISTR_continuous_Lakes_irrig_waterUses_MODIrrRes_Pd

def cal_MISDC_SEMIDISTR_continuous_Lakes(input, BAS_PAR, EBRR_BASPAR, sez_outlet, bas_check, ID_bas_app, Lake, ID_irr, X_ini=None):
    NPAR = 12
    Nbas = BAS_PAR[0]  

    # Initialize X_ini if not provided
    if X_ini is None:
        X_ini = np.ones(NPAR) * 0.5

    # Define parameter bounds as numpy arrays
    lower_bounds = np.array([
        0.1,    # W_p
        300,    # W_max2
        2.0,    # m2
        0.10,   # Ks
        0.5,    # gamma1
        0.4,    # Kc
        1.0,    # alpha
        0.1/24, # Cm
        5.0,    # m22
        0.01,   # Ks2
        1,      # C
        1       # D
    ])

    upper_bounds = np.array([
        0.9,    # W_p
        4000,   # W_max2
        10.0,   # m2
        20.0,   # Ks
        3.5,    # gamma1
        2.0,    # Kc
        15.0,   # alpha
        3.0,    # Cm
        20.0,   # m22
        45.0,   # Ks2
        60,     # C
        30      # D
    ])

    bounds = list(zip(lower_bounds, upper_bounds))

    def objective(X_0 ,input, BAS_PAR, EBRR_BASPAR, sez_outlet, bas_check, ID_bas_app, Lake, ID_irr):
        """
        Objective function for calibration
        """
        try:
            X=X_0
            PAR = np.tile(X, (Nbas, 1)).T
            NS, KGE_sez, KGE_out,_,_,_,_,_,_,_ = MISD_SEMIDISTR_continuous_Lakes_irrig_waterUses_MODIrrRes_Pd(
                input, BAS_PAR, EBRR_BASPAR, PAR, sez_outlet, 
                bas_check, ID_bas_app, Lake, ID_irr, 1, include_irrigation=True, include_reservoirs=True
            )
            return 1 - KGE_out
        except Exception as e:
            print(f"Error in objective function: {str(e)}")
            return 1e10
    try:
        # Normalize initial parameters
        X_ini_norm = X_ini* (upper_bounds - lower_bounds)+lower_bounds
        options = {'ftol': 1e-6, 'maxiter': 100, 'disp': False}    
        result = minimize(objective, X_ini_norm, args=(input, BAS_PAR, EBRR_BASPAR, sez_outlet, bas_check, ID_bas_app, Lake, ID_irr),
                          method='TNC', bounds=bounds, options=options)
        X_OPT = result.x
        PAR = np.tile(X_OPT, (Nbas, 1)).T
        return PAR
    except Exception as e:
        X_default = lower_bounds +(upper_bounds-lower_bounds ) / 2
        return np.tile(X_default, (Nbas, 1)).T

def convert_adim(X_0, NsezCheck):
    """
    Convert normalized parameters to actual parameter values
    """
    # Lower and upper bounds for parameters
    LOW = np.array([
        0.1,    # W_p
        300,    # W_max2
        2.0,    # m2
        0.10,   # Ks
        0.5,    # gamma1
        0.4,    # Kc
        1.0,    # alpha
        0.1/24, # Cm
        5.0,    # m22
        0.01,   # Ks2
        1,      # C
        1       # D
    ])

    UP = np.array([
        0.9,    # W_p
        4000,   # W_max2
        10.0,   # m2
        20.0,   # Ks
        3.5,    # gamma1
        2.0,    # Kc
        15.0,   # alpha
        3,      # Cm
        20.0,   # m22
        45.0,   # Ks2
        60,     # C
        30      # D
    ])

    X_0 = np.asarray(X_0)
    if X_0.ndim == 1:
        X_0 = X_0.reshape(-1, 1)
    LOW = np.tile(LOW.reshape(-1, 1), (1, NsezCheck))
    UP = np.tile(UP.reshape(-1, 1), (1, NsezCheck))
    X = LOW + (UP - LOW) * X_0
    return X



