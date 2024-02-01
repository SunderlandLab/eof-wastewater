# WRF wastewater project
#
# Copyright (c) 2023 Bridger Ruyle <bruyle@carnegiescience.edu>
# Sunderland Lab
# Harvard University 

"""Determine the relationship between EOF discharge and population"""
import numpy as np
import pandas as pd 
import pymc3 as pm
import arviz as az
from scipy.stats import gmean
import statsmodels.api as sm


def main():
    #read in wrf data 
    df = pd.read_csv('github/input-data.csv')
    df['Discharge_meas'] = df['Discharge (L/d)']*df['EOF (nM)']/1E9 #mol/day
    df.sort_values('Population', inplace = True)

    #log(discharge) = log_beta + alpha*log(population)
    #perform ols to get central estimate of log_beta and alpha coefficients as priors 
    y = np.log(df['Discharge_meas'])
    X = np.log(df['Population'])
    X = sm.add_constant(X)
    lm = sm.OLS(y, X).fit()

    #perform Bayesian linear regression to get posterior distribution of log_beta and alpha
    with pm.Model() as model:
        log_beta = pm.Normal('intercept', lm.params[0], 1)
        alpha = pm.Normal('slope', lm.params[1], 1)
        s = pm.Normal('error', 1)

        population = pm.Data('population', np.log(df['Population']))
        start = pm.find_MAP()
        obs = pm.Normal('discharge', log_beta + alpha*population, s, observed = np.log(df['Discharge_meas']))
 
        trace = pm.sample(5000, tune = 1000, chains = 4, return_inferencedata = False, target_accept = 0.9, cores = 4)

    #sample posterior predictive
    with model:
        pcc = pm.sample_posterior_predictive(trace, random_seed = 1876)

    # #descriptive statistics of posterior distribution
    summary = az.summary(trace)
    alpha_mean = summary['mean']['slope']; alpha_std = summary['sd']['slope']
    log_beta_mean = summary['mean']['intercept']; log_beta_std = summary['sd']['slope']
    beta_mean = np.exp(log_beta_mean+log_beta_std**2/2); beta_std = np.sqrt((np.exp(log_beta_std**2)-1)*np.exp(2*log_beta_mean+log_beta_std**2))

    #apply results to large POTWs in the Clean Watersheds Needs Survey (CWNS)
    cwns = pd.read_excel('github/SUMMARY_POPULATION.xlsx')
    cwns= cwns[cwns['PRES_RES_RECEIVING_COLLCTN'] >= 10000]

    #switch Bayesian linear regression data to CWNS data and predict discharges
    pm.set_data({'population':np.log(cwns['PRES_RES_RECEIVING_COLLCTN'])}, model = model)
    with model:
        wwtp_prediction = pm.sample_posterior_predictive(trace, random_seed = 1876)

    #calculate total discharges for large POTWs: mean and 90th CI
    total_discharge = np.sum(np.exp(wwtp_prediction['discharge']), axis = 1)
    total_discharge_mean = 365*gmean(total_discharge)
    total_discharge_lb = 365*np.quantile(total_discharge, 0.05)
    total_discharge_ub = 365*np.quantile(total_discharge, 0.95)

    #save results
    optimized_coefs = pd.DataFrame({'alpha': [alpha_mean, alpha_std],
                                    'beta': [beta_mean, beta_std]},
                                    columns = ['alpha', 'beta'], index = ['mean', 'st.dev'])
    optimized_coefs.to_csv('github/regression-res.csv')

    total_discharge_df = pd.DataFrame({'mean discharge (mol/yr)': [total_discharge_mean], '5th percentile (mol/yr)': [total_discharge_lb],
                                       '95th percentile (mol/yr)':[total_discharge_ub]})
    total_discharge_df.to_csv('github/discharge-res.csv', index = False)
    
if __name__ == '__main__':
    main()