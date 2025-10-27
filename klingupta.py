import numpy as np

def klinggupta(simulated,observed):
    observed = np.asarray(observed).flatten()
    simulated = np.asarray(simulated).flatten()
    observed = np.array(observed)
    simulated = np.array(simulated)
    mask = ~np.logical_or((np.isnan(observed)),(np.isnan(simulated)))
    observed = observed[mask]
    simulated = simulated[mask]
    
    def corrcoef(x, y):
        x = np.array(x)
        y = np.array(y)
        mask = ~(np.isnan(x))
        x_valid = x[mask]
        y_valid = y
        if len(x_valid) > 0:
            return np.corrcoef(x_valid, y_valid)[0, 1]
        else:
            return np.nan

    # calculate correlation coefficient (r)
    r = corrcoef(observed, simulated)

    # Calculate bias ratio (β)
    beta = np.mean(simulated) / np.mean(observed)

    # Calculate variability ratio (γ)
    cv_obs = np.std(observed) / np.mean(observed)
    cv_sim = np.std(simulated) / np.mean(simulated)
    gamma = cv_sim / cv_obs
    relvar=np.std(simulated) / np.std(observed)

    # Calculate modified KGE
    kge = 1 - np.sqrt(((r - 1)**2) + ((beta - 1)**2) + ((relvar - 1)**2))
    kge_ = 1 - np.sqrt(((r - 1)**2) + ((beta - 1)**2) + ((gamma - 1)**2))


    return kge, kge_, relvar, beta