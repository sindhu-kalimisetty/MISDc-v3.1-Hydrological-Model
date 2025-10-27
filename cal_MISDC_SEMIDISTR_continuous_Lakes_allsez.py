import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from MISD_SEMIDISTR_continuous_model import MISD_SEMIDISTR_continuous_Lakes_irrig_waterUses_MODIrrRes_Pd

def objective_function(X_0, input, BAS_PAR, EBRR_BASPAR, sez_outlet, bas_check, ID_bas_app, Lake, ID_irr, ID, X_optmod):
        try:            
            X = convert_adim(X_0, 1)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            PAR = X_optmod.copy()
            PAR[:, ID] = np.tile(X, (1, len(ID)))
            NS, KGE_sez, KGE_out,_,_,_,_,_,_,_ = MISD_SEMIDISTR_continuous_Lakes_irrig_waterUses_MODIrrRes_Pd(input, BAS_PAR, EBRR_BASPAR, PAR, sez_outlet, bas_check, ID_bas_app, Lake, ID_irr, 1, include_irrigation=True, include_reservoirs=True)                
            if np.isnan(NS) or NS < -0.5:
                return 1e10   
            return 1 - NS
        except Exception:
            return 1e10
        
def convert_adim(X_0, NsezCheck):
        X_0 = np.asarray(X_0).ravel()
        LOW = np.array([0.1, 300, 2.0, 0.10, 0.5, 0.4, 1.0, 0.1/24, 5.0, 0.01, 1, 1])
        UP = np.array([0.9, 4000, 10.0, 20.0, 3.5, 2.0, 15.0, 3, 20.0, 45.0, 60, 30])
        LOW = LOW.reshape(-1, 1)
        UP = UP.reshape(-1, 1)
        X_0 = X_0.reshape(-1, 1)
        X = LOW + (UP - LOW) * X_0
        return X

def cal_MISDC_SEMIDISTR_continuous_Lakes_allsez(input, BAS_PAR, EBRR_BASPAR, sez_outlet, bas_check, ID_bas_app, Lake, ID_irr, ID, X_optmod, X_ini=None):
    NPAR = 12
    Nbas = BAS_PAR[0]  
    if X_ini is None:
        X_ini = np.ones(NPAR) * 0.5
    bounds = [
        (0.0, 1.0) for _ in range(NPAR)  
    ]
    args = (input, BAS_PAR, EBRR_BASPAR, sez_outlet, bas_check, ID_bas_app, Lake, ID_irr, ID, X_optmod)
    
    # Run differential evolution
    result = differential_evolution(
        objective_function,
        bounds,
        args=args,
        strategy='best1bin',
        maxiter=40,
        popsize=15,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True,
        updating='deferred',
        workers=-1  
    )
    
    RES = result.x
    FVAL = result.fun
    X_OPT = convert_adim(RES, 1)
    if X_OPT.ndim == 1:
        X_OPT = X_OPT.reshape(-1, 1)
    PAR = X_optmod.copy()
    PAR[:, ID] = np.tile(X_OPT, (1,len(ID)))
    return PAR
