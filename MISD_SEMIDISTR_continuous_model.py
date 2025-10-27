import time
timing_data = {}
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from klingupta import klinggupta
import math
# from hydrostats.metrics import r_squared
from Water_use import calculate_water_use
from scipy.stats import pearsonr
import time
import functools

def MISD_SEMIDISTR_continuous_Lakes_irrig_waterUses_MODIrrRes_Pd(input_data, BAS_PAR, EBRR_BASPAR, PAR, sez_outlet, bas_check, ID_bas_app, Lake, ID_irr, FIG, include_irrigation=True, include_reservoirs=True, W2ini=None,):

    # Load basin parameters
    Nbas = BAS_PAR[0]  # number of subcatchments
    Nsez = BAS_PAR[1]  # number of sections
    Ninf = BAS_PAR[2]  # number of upstream inflows

    basin_data = input_data
    data = basin_data[0]['dates']
    
    # convert to hours
    clean_data = pd.Series(data)[pd.notna(data)]
    differences_hours = clean_data.diff().dt.total_seconds() / 3600
    delta_T = round(differences_hours.mean() * 10000) / 10000
    
    # Morphological data
    DIST = EBRR_BASPAR[:, 1:Nsez+1]    # catchment distance to outlet sections
    Ab = EBRR_BASPAR[:, Nsez+1][Ninf:]  # catchments area
    A_DD = EBRR_BASPAR[:, Nsez+2][Ninf:]  # catchments type (1: concentrator, 2: distributed)
    
    # Initialization
    M = basin_data[0]['dates'].shape[0]
    QQsim = np.zeros((M, Nbas))    # catchments runoff (mm/delta_T)
    QQQsim = np.zeros((M, Nbas+Ninf, Nsez))  # sections runoff (mm/delta_T)
    QQQBF = np.zeros((M, Nbas+Ninf, Nsez))  # sections baseflow (mm/delta_T)
    Stor = np.zeros((M, Nbas))    # storage Lakes (Mcm/delta_T)
    # Initialize result arrays
    WW_all = [None] * Nbas  # List to store WW for each basin
    WW2_all = [None] * Nbas  # List to store WW2 for each basin
    VorrIrr_all = [None] * Nbas  # List to store VolIRR for each basin
    E_all = [None] * Nbas  # List to store E for each basin
    SWE_all = [None] * Nbas  # List to store SWE_pack for each basin
    BFmean_all = np.zeros((M, Nbas))  # Array to store BF means
    peffmean_all = np.zeros((M, Nbas))  # Array to store peff means
    unMeetDem_sum = np.zeros((M, Nbas))  # Array to store unmet demand sums
    VolIrr_sum = np.zeros((M, Nbas))  # Array to store irrigation volume sums
    VolIrr_sumDisc = np.zeros((M, Nbas))  # Array to store distributed irrigation volume sums
    QQBF = np.zeros((M, Nbas))  # Array to store baseflow
    QBF1 = np.zeros((M, Nsez))  # Array to store section baseflow
    Qsim1 = np.zeros((M, Nsez))  # Array to store section discharge
    KGE_sez = np.zeros(Nsez)  # Array to store KGE values
    NS_sez = np.zeros(Nsez)  # Array to store NS values
    QunMeet = np.zeros(Nbas)  # Array to store unmet demand

    # Fixed parameters
    W_max = 200    # FIXED WATER CAPACITY 1st LAYER

    # Helper functions
    def GIUH(gamma, Ab, dt, deltaT):
        """Calculate Geomorphological Instantaneous Unit Hydrograph"""
        Lag = (gamma * 1.19 * Ab**0.33) / deltaT
        hp = 0.8 / Lag
        # Load GIUH data 
        data = np.loadtxt('GIUH')  
        t = data[:, 0] * Lag
        IUH_0 = data[:, 1] * hp
        ti = np.arange(0, np.max(t), dt)
        IUH = np.interp(ti, t, IUH_0)
        IUH = IUH / (np.sum(IUH) * dt)
        return IUH

    def NASH(gamma, Ab, dt, deltaT, n):
        """Calculate Nash Instantaneous Unit Hydrograph"""
        K = (gamma * 1.19 * Ab**0.33) / deltaT
        Tmax = 100  # hours
        time = np.arange(0.00001, Tmax, dt)
        IUH = ((time/K)**(n-1) * np.exp(-time/K)) / (math.factorial(n-1) * K)
        IUH = IUH / (np.sum(IUH) * dt)
        return IUH

    def hayami(dt, L, C, D, deltaT):
        """Calculate Hayami function (diffusive routing)"""
        C = C * deltaT
        D = D * deltaT
        Tmax = 100  # hours
        tt = np.arange(0.00001, Tmax, dt)
        g = (L / np.sqrt(4 * np.pi * D * tt**3)) * np.exp(-((L - C*tt)**2) / (4 * D * tt))
        g = g / (np.sum(g) * dt)
        return g
    
    def snow_model(precipitation, temperature, temp_min, temp_max, Cm):
        """Snow accumulation-melting model"""
        rainfall = np.zeros_like(precipitation)
        snowfall = np.zeros_like(precipitation)
        SWE_snowpack = np.zeros_like(precipitation)
        SWE_melting = np.zeros_like(precipitation)
        
        ID1=np.logical_or(np.isnan(precipitation),(np.isnan(temperature)))
        ID2=temperature<=temp_min
        ID3=temperature>=temp_max
        ID4=np.logical_and(temperature>temp_min,temperature<temp_max)
        
        rainfall[ID1]=np.nan
        snowfall[ID1]=np.nan
        rainfall[ID2]=0
        snowfall[ID2]=precipitation[ID2]
        rainfall[ID3]=precipitation[ID3]
        snowfall[ID3]=0
        SWE_melting[ID3]=Cm * (temperature[ID3] - temp_max)
        rainfall[ID4]=precipitation[ID4]*((temperature[ID4]-temp_min)/(temp_max-temp_min))
        snowfall[ID4]=precipitation[ID4]-rainfall[ID4]
        SWE_melting[0,:]=0
        SWE_snowpack[0,:]=snowfall[0,:]
        
        for i in range(1, precipitation.shape[0]):
            SWE_snowpack[i,:]=np.nansum([SWE_snowpack[i-1,:],snowfall[i,:],-1*SWE_melting[i,:]],0)
            ID=SWE_snowpack[i,:]<0
            SWE_melting[i,ID]=SWE_snowpack[i-1,ID]
            SWE_snowpack[i,ID]=0
            
        SWE_snowpack[ID1]=np.nan


        return rainfall, SWE_melting, SWE_snowpack

    def burek2013_func(C, time_series, inflow_obs):
        """
        Reservoir function implementation (Burek et al. 2013)
        """
        # Calculate mean inflow
        Qin_mean = np.nanmean(inflow_obs)

        # Initialize storage and outflow arrays
        S = np.full((len(time_series), 1), np.nan)
        S[0] = 0.8 * C  # Initial storage at 80% capacity
        Qres = np.zeros((len(time_series),1))

        # Define reservoir parameters
        Qmin = 0.4 * Qin_mean  # Minimum flow
        Qnd = 2.0 * Qin_mean   # Normal discharge
        Qnorm = Qin_mean       # Normal flow

        # Define storage thresholds
        Lc = 0.1  # Critical level
        Lf = 0.9  # Flood level
        Ln = Lf - Lc  # Normal level

        iter_count = 0
        for t in range(1, len(time_series)):
            if isinstance(time_series[t] - time_series[t - 1], np.timedelta64):
                dt = (time_series[t] - time_series[t - 1]) / np.timedelta64(1, "s")
            else:
                dt = (time_series[t] - time_series[t - 1]).total_seconds()

            if t==1:
                F = S[t-1] / C
                if F <= 2 * Lc:
                    Qres[t-1,0] = min(Qmin, (1/dt) * F * S[t-1])
                elif F > 2 * Lc and F <= Ln:
                    Qres[t-1,0] = Qmin + (Qnorm - Qmin) * ((F - 2*Lc)/(Ln - 2*Lc))
                elif F > Ln and F <= Lf:
                    Qres[t-1,0] = Qnorm + ((F - Ln)/(Lf - Ln)) * max([(inflow_obs[t-1] - Qnorm), (Qnd - Qnorm)])
                elif F > Lf:
                    Qres[t-1,0] = max([(F - Lf)/dt * S[t-1], Qnd])

            asd=0
            n=0
            Qres[t] = Qres[t-1]
            while asd==0:
                n=n+1
                iter_count += 1
                S[t] = S[t-1] + ((inflow_obs[t] + inflow_obs[t-1])/2) * dt - ((Qres[t] + Qres[t-1])/2) * dt
                F = S[t] / C
                if F <= 2 * Lc:
                    Qres[t,0] = min(Qmin, (1/dt) * F * S[t])
                elif F >= 2 * Lc and F < Ln:
                    Qres[t,0] = Qmin + (Qnorm - Qmin) * ((F - 2*Lc)/(Ln - 2*Lc))
                elif F > Ln and F <= Lf:
                    Qres[t,0] = Qnorm + ((F - Ln)/(Lf - Ln)) * max([(inflow_obs[t] - Qnorm), (Qnd - Qnorm)])
                elif F > Lf:
                    Qres[t,0] = max([(F - Lf)/dt * S[t], Qnd])
                
                S0 = S[t-1] + ((inflow_obs[t] + inflow_obs[t-1])/2) * dt - ((Qres[t] + Qres[t-1])/2) * dt
                if np.allclose(S0, S[t], rtol=1e-3, atol=1e-3):  
                    asd=1
                else:
                    S[t] = S0

                if n>1000:
                    break
        return Qres, S
    
    # Main processing loop
    for i in range(Nbas):
        start_time = time.time()
        # Extract parameters
        W_p = PAR[0,i]      # initial conditions, fraction of W_max (0-1)
        W_max2 = PAR[1,i]   # total water capacity of 2nd layer
        m2 = PAR[2,i]       # exponent of drainage for 1st layer
        Ks = PAR[3,i]       # hydraulic conductivity for 1st layer
        gamma = PAR[4,i]    # coefficient lag-time relationship
        Kc = PAR[5,i]       # parameter of potential evapotranspiration
        alpha = PAR[6,i]    # exponent runoff
        Cm = PAR[7,i]       # Snow module parameter degree-day
        m22 = PAR[8,i]      # exponent of drainage for 2nd layer
        Ks2 = PAR[9,i]      # hydraulic conductivity for 2nd layer
        C = PAR[10,i]       # Celerity
        Diff = PAR[11,i]    # Diffusivity

        dt = 0.2    # Computation time step in hours
        Ks = Ks * delta_T    # Convert to mm/delta_T
        Ks2 = Ks2 * delta_T  # Convert to mm/delta_T

        # Extract basin data
        D = basin_data[i]['dates']

        PIO_ = basin_data[i]['rainfall']
        TEMPER = basin_data[i]['temperature']
        Q = basin_data[i]['discharge']
        # WUse = basin_data[i]['water_use']    

        # extract months  
        MESE = np.array([pd.Timestamp(d).month if not pd.isna(d) else np.nan for d in D])
        MESE = MESE.reshape(-1, 1)

        # Snow Module
        PIO, SWE, SWE_pack = snow_model(PIO_, TEMPER, -0.5, 0.5, Cm)

        WUse = calculate_water_use(
        time_period=D,
        sector='both',
        base_demand_civil=0.133,
        base_demand_industrial=0.089,
        seasonal_factor=True,
        # daily_pattern=True,
        climate_zone='temperate',
        industrial_type='mixed'
        )

        if WUse.shape[1] != PIO.shape[1]:
            WUse = np.tile(WUse, (1, PIO.shape[1]))
       
        # Potential Evapotranspiration calculation
        L = np.array([0.21, 0.22, 0.23, 0.28, 0.30, 0.31,
                    0.30, 0.29, 0.27, 0.25, 0.22, 0.20])
        Ka = 1.26
        EPOT = np.where(TEMPER > 0, (Kc * (Ka * L[MESE-1] * (0.46 * TEMPER + 8) - 2)) / (24/delta_T),0)
                
        # Initialize arrays
        BF = np.zeros_like(PIO)
        QS = np.full_like(PIO, np.nan)
        VolIRR_distr = np.zeros_like(PIO)
        W = np.zeros_like(PIO)
        W2 = np.zeros_like(PIO)
        WW = np.zeros_like(PIO)
        WW2 = np.zeros_like(PIO)
        IE = np.zeros_like(PIO)
        PERC = np.zeros_like(PIO)
        PERC2 = np.zeros_like(PIO)
        VolIRR = np.zeros_like(PIO)
        water_need = np.zeros_like(PIO)
        unMeetDem = np.zeros_like(PIO)
        E = np.zeros_like(PIO)
        SE = np.zeros_like(PIO)
        SE2 = np.zeros_like(PIO)

        ID_irr_bas = ID_irr[0][i]
            
        W[0, :] = W_p * W_max
        W2[0, :] = W_p * W_max2

        for t in range(M):
            # Water balance calculations
            if t > 0:
                W[t, :] = W[t-1, :]
                W2[t, :] = W2[t-1, :]

            # Calculate infiltration excess
            IE[t, :] = PIO[t, :] * ((W[t, :]/W_max)**alpha)

            # Calculate evaporation
            E[t, :] = EPOT[t, :] * W[t, :]/W_max

            # Calculate percolation
            ID= W2[t, :] < W_max2
            PERC[t, ID] = Ks * (W[t, ID]/W_max)**(m2)    
            
            PERC2[t, :] = Ks2 * (W2[t, :]/W_max2)**(m22)

            ID=PERC[t, :] > 0.75 * W_max
            PERC[t, ID] = 0.75 * W_max

            # Update water content
            W[t, :] = W[t, :] + (PIO[t, :] - IE[t, :] - PERC[t, :] - E[t, :]) + SWE[t, :]
            W[t, W[t, :]<0] = 0
            
            W2[t, :] = W2[t, :] + PERC[t, :] - PERC2[t, :] - WUse[t, :]
            W2[t, W2[t, :]<0] = 0
            
            ID= W[t, :] >= W_max
            SE[t, ID] = W[t, ID] - W_max
            W[t, ID] = W_max
            
            ID=W2[t, :] >= W_max2
            SE2[t,ID] = W2[t, ID] - W_max2
            W2[t, ID] = W_max2
            
            if include_irrigation and MESE[t] in  [4, 5, 6, 7, 8, 9]:
                DISTR_Irr = np.where(ID_irr_bas.flatten() == 1)[0]
                DISTR_Irr=DISTR_Irr[DISTR_Irr<W.shape[1]]
                
                Thres_irr = 0.55
                Thres_ini = 0.35

                ID=DISTR_Irr[W[t,DISTR_Irr]<Thres_ini * W_max]  

                water_need[t, ID] = Thres_irr - (W[t, ID]/W_max)
                
                ID2=W2[t, ID] > water_need[t, ID] * W_max
                W[t, ID[ID2]] = Thres_irr * W_max
                W2[t, ID[ID2]] -= 0.17 * water_need[t, ID[ID2]] * W_max

                
                W[t, ID[~ID2]] += 0.1 * water_need[t, ID[~ID2]]
                W2[t, ID[~ID2]] -= 0.1 * 0.17 * water_need[t, ID[~ID2]] * W_max

                VolIRR[t, ID] = water_need[t, ID] * W_max

                # Distribute irrigation volume
                end_idx = min(t+12, VolIRR_distr.shape[0])
                VolIRR_distr[t:end_idx, :] += np.nansum(VolIRR[t, ID]/((end_idx-t) * PIO.shape[1]))
                
            # Ensure non-negative values
            W[t, W[t, :]<0] = 0
            W2[t, W2[t, :]<0] = 0
            
            WW[t, :] = W[t, :]/W_max
            WW2[t, :] = W2[t, :]/W_max2

            # Calculate runoff components
            BF[t, :] = Ks2 * ((W[t, :] + W2[t, :])/(W_max + W_max2))**(m22)
            QS[t, :] = IE[t, :] + SE[t, :] + SE2[t, :] - 0.80 * VolIRR_distr[t, :]

            # Track unmet demand
            ID=QS[t, :] + BF[t, :] < 0
            unMeetDem[t, ID] = (IE[t, ID] + SE[t, ID] + SE2[t, ID])/0.80
          
        WW_all[i] = WW
        WW2_all[i] = WW2
        VorrIrr_all[i] = VolIRR
        E_all[i] = E
        SWE_all[i] = SWE_pack
            
        # Calculate means
        peffmean = np.nanmean(QS, axis=1)
        BF_mean = np.nanmean(BF, axis=1)
        BFmean_all[:, i] = BF_mean
        peffmean_all[:, i] = peffmean
        unMeetDem_sum[:, i] = np.nansum(unMeetDem, axis=1)
        VolIrr_sum[:, i] = np.nanmean(VolIRR, axis=1)

        # Convolution calculations
        if Ab[i] > 0.0:
            if A_DD[i] == 1:
                IUH = GIUH(gamma, Ab[i], dt, delta_T) * dt        
            else:
                IUH = NASH(2 * gamma, Ab[i], dt, delta_T, 1) * dt                              
            # Interpolation and convolution
            t_points = np.arange(1, M+1)
            t_interp = np.arange(1, M+1, dt)
            t_interp = t_interp[t_interp <= M]  
            peffint = np.interp(t_interp, t_points, peffmean)
            BFint = np.interp(t_interp, t_points, BF_mean)
            temp1 = np.convolve(IUH, peffint)
            temp2 = np.convolve(NASH(2 * gamma, Ab[i], dt, delta_T, 1) * dt, BFint)
            step = round(1/dt)
            QQBF[:, i] = temp2[::step][:M]
            QQsim[:, i] = temp1[::step][:M] + temp2[::step][:M]
        else:
            QQBF[:, i] = BF_mean
            QQsim[:, i] = peffmean + BF_mean

        # Convolution (Hayami)
        for j in range(Nsez):
            if DIST[i + Ninf, j] > 0:
                g = hayami(dt, DIST[i + Ninf, j], C, Diff, delta_T) * dt

                # Interpolation for dt time step
                x = np.arange(1, M + 1)
                x_new = np.arange(1, M + 1, dt)
                x_new = x_new[x_new <= M] 
                QQsimint = np.interp(x_new, x, QQsim[:, i]).T
                QQBFint = np.interp(x_new, x, QQBF[:, i]).T

                # Convolution
                temp1 = np.convolve(g, QQsimint)
                temp2 = np.convolve(g, QQBFint)

                step = round(1/dt)
                QQQBF[:, i + Ninf, j] = temp2[::step][:M]  # in delta_T
                QQQsim[:, i + Ninf, j] = temp1[::step][:M]

            elif DIST[i + Ninf, j] == 0:
                QQQBF[:, i + Ninf, j] = QQBF[:, i]
                QQQsim[:, i + Ninf, j] = QQsim[:, i]

            elif DIST[i + Ninf, j] < 0:
                QQQBF[:, i + Ninf, j] = np.zeros(M)
                QQQsim[:, i + Ninf, j] = np.zeros(M)

    for i in range(Nbas):
        QQBF[:, i] *= (Ab[i]/delta_T/3.6)
        QQsim[:, i] *= (Ab[i]/delta_T/3.6)

    # Calculate sections discharge in m^3/s
    for i in range(Nbas):
        for j in range(Nsez):
            QQQBF[:, i+Ninf, j] *= (Ab[i]/delta_T/3.6)
            QQQsim[:, i+Ninf, j] *= (Ab[i]/delta_T/3.6)
            
        VolIrr_sumDisc[:, i] = VolIrr_sum[:, i] * (Ab[i]/delta_T/3.6)

    # Initialize lake simulation array
    QQQsim_outlake = np.zeros((M, Nbas))
    if include_reservoirs and Lake.shape[0] > 0:
        for l in range(Lake.shape[0]):
            lake_idx = int(Lake[l,0])  
            ID = np.where(EBRR_BASPAR[:, lake_idx] != -1)[0]
            lake_inflow = (np.nansum(QQQsim[:, ID, lake_idx-1], axis=1) - 
                        0.03 * Lake[l,1]/np.sum(Lake[:,1]) * np.nansum(VolIrr_sumDisc[:, ID], axis=1))
            outlake, Stor_lake = burek2013_func(
                Lake[l,1], 
                basin_data[0]['dates'],  
                lake_inflow
            )
            QQQsim_outlake[:, lake_idx-1]=outlake.flatten()
            Stor[:, lake_idx-1]=Stor_lake.flatten()
            sez_contr = np.where(EBRR_BASPAR[lake_idx-1, 1:-2] != -1)[0]
            IDdel = np.isin(ID, lake_idx-1)
            ID = ID[~IDdel]
  
            # Update discharge values
            for j in range(len(sez_contr)):
                QQQsim[:, lake_idx-1, sez_contr[j]] = QQQsim_outlake[:, lake_idx-1]
                QQQsim[:, ID, sez_contr[j]] = np.zeros((QQQsim.shape[0], len(ID)))

    for j in range(Nsez):
        for i in range(Ninf):
            if DIST[i,j] > 0.0:
                g = hayami(dt, DIST[i,j], C, Diff, delta_T) * dt
                x = np.arange(1, M + 1)
                x_new = np.arange(1, M + 1, dt)
                QQsimint = np.interp(x_new, x, QM[:, i])
                temp1 = np.convolve(g, QQsimint)
                step = round(1/dt)
                QQQsim[:, i, j] = temp1[::step][:M]

    QBF1 = np.nansum(QQQBF, axis=1)
    Qsim1 = np.nansum(QQQsim, axis=1)
    obs_data=np.array([np.squeeze(bb['discharge']) for bb in basin_data]).T
    
    for j in range(Nsez):
        ID=np.where(~np.logical_or(np.isnan(obs_data[:,j]),np.isnan(Qsim1[:,j])))[0]
        if np.size(ID)!=0:
            KGE_sez[j]=1-((pearsonr(obs_data[ID,j],Qsim1[ID,j])[0]-1)**2+(np.nanmean(Qsim1[ID,j])/np.nanmean(obs_data[ID,j])-1)**2+(np.nanstd(Qsim1[ID,j])/np.nanstd(obs_data[ID,j])-1)**2)**0.5

    NS_sez = 1 - (np.nansum((Qsim1 - obs_data)**2,axis=0) / 
                     np.nansum((obs_data - np.nanmean(obs_data,axis=0))**2,axis=0))

    # Extract simulation results
    Qsim = Qsim1[:, sez_outlet-1]
    QBF_out = np.hstack((QBF1, QQBF))
    Qsim_out = np.hstack((Qsim1, QQsim))

    # Calculate unmet demand
    for ll in range(BAS_PAR[0]):
        mask = Qsim_out[:, ll] < 0
        QunMeet[ll] = np.nansum(np.abs(Qsim_out[mask, ll]))

    # Set negative values to zero
    Qsim_out[Qsim_out < 0] = 0

    # Get observed discharge
    Qobs = basin_data[bas_check-1]['discharge']

    #Nash-Sutcliffe Efficiency
    NS = 1-(np.nansum((Qsim - Qobs)**2)/np.nansum((Qobs - np.nanmean(Qobs))**2))

    #Kling-Gupta Efficiency
    KGE_out = klinggupta(Qsim, Qobs)
    
    return NS, KGE_sez, KGE_out, Qsim_out, Stor, WW_all, WW2_all, VorrIrr_all, SWE_all, E_all









