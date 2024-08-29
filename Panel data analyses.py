#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:35:34 2024

@author: thierrycoulibaly
"""
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects
import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import pypandoc
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.linalg as la  
from scipy import stats
from linearmodels.iv import IV2SLS, compare
import numpy as np 
import os

# Changing directories
#os.chdir('/Users/thierrycoulibaly/Library/CloudStorage/OneDrive-KyushuUniversity/Work/Jilan')
os.chdir('C:/Users/Coulibaly Yerema/OneDrive - Kyushu University/Work/Jilan')

##############################################################################
############################ Main Panel data analyses ########################
##############################################################################
# Importing the data
df_copy = pd.read_csv('Panel Data all countries.csv')


# Replacing the values with the forcaseted ones
df = df_copy

df = df.dropna()

# Cleaning the dataset
# Adding a very small value to ensure the variable buses can transform to log
df['Log_Buses'] = np.log(df['Buses'] + 0.0001)
df['Log_PM25'] = np.log(df['PM25 microgram per cubic metter'] + 0.0001)
df['Log_urban_areas'] = np.log(df['LC_urban_areas'])
df['Log_green_area'] = np.log(df['Green_area'] + 0.0001)
df['Log_pop'] = np.log(df['Pop density heads per Sqkm'] + 0.0001)
df['Log_cars'] = np.log(df['Cars'] + 0.0001)
df['Log_trucks'] = np.log(df['Trucks'] + 0.0001)
df['Log_motorcycles'] = np.log(df['Motorcycles'] + 0.0001)


# Figure on the reslationship between urban areas and buses
ax = sns.regplot(data=df[df.Log_Buses > -5], x= 'Log_urban_areas', y='Log_Buses')
ax.set(xlabel='Log(Urban areas)',
       ylabel='Log(Buses)',
       title='Relationship between the instrument and the endogenous variable')
plt.show()
fig = ax.get_figure()
fig.savefig('test.png', dpi=600)

# Summary statistics
Summary = df.describe(percentiles=[0.5]).T
print(Summary)
Summary.to_excel("Summary Statistics.xlsx")

# Histogram of the variables
columns_for_hist = ['Log_PM25', 'Log_Buses','Buses','Log_urban_areas',
                    'Average_wind_speed','Average_precipitation','Maximum_temperature',
                    'Log_motorcycles','Green_area','Log_green_area','Log_trucks',
                    'Log_pop', 'Log_cars', 'Log_motorcycles']
histograms_of_data = df.hist(column = columns_for_hist, figsize=(20, 20), ec="k")
fig = histograms_of_data[0, 0].get_figure()
fig.savefig('Histograms_figure.png')

## Setting the data for regression analyses / defining the variable
# Changing the variable type to ensure thy will considered as dummy in the regression
df = df.reset_index()
df['Id'] = df['Id'].astype('category')
df['year'] = df['year'].astype('category')

# Adding the constant variable
df = sm.add_constant(df)

# Defining the controls variables
X = df[['const','Average_wind_speed','Average_precipitation','Maximum_temperature',
         'Log_motorcycles','Log_green_area','Id','Log_cars','Log_trucks',
         ]]

# Defining the variables of interest
y = df['Log_PM25']
endog_var = df['Log_Buses']
Instrument = df['Log_urban_areas']


######################### Running the regressions ##############################
########
## Performing an OLS (panel data fixed effect regression) to compare results with 2SLS
X_OLS = df[['const','Log_Buses','Average_wind_speed','Average_precipitation','Maximum_temperature',
        'Log_cars','Log_trucks', 'Log_motorcycles', 'Log_green_area', 'Id']]

########
# Runing the OLS with no control variable
Two_vars_OLS = IV2SLS(y, df[['const', 'Log_Buses', 'Id']], None, None).fit()
print(Two_vars_OLS.summary)

########
# Runing the OLS with greeneries as control variable
Three_vars_OLS= IV2SLS(y, df[['const', 'Log_green_area', 'Log_Buses', 'Id']], None, None).fit()
print(Three_vars_OLS.summary)

########
# Runing the OLS with all control variables
Full_OLS = IV2SLS(y, X_OLS, None, None).fit()
print(Full_OLS.summary)

################
# Runing the 2SLS with no control variable
Two_vars_2SLS = IV2SLS(y, df[['const', 'Id']], endog_var, Instrument).fit()
print(Two_vars_2SLS.summary)
#print(Two_vars_2SLS.first_stage)

# Wald test for exogeneity of the endogeneous variable (to know whether it is necessary to run 2SLS)
#print(Two_vars_2SLS.wooldridge_regression)

########
# Runing the 2SLS with greeneries as control variable
Three_vars_2SLS = IV2SLS(y, df[['const', 'Id', 'Log_green_area']], endog_var, Instrument).fit()
print(Three_vars_2SLS.summary)
#print(Three_vars_2SLS.first_stage)

# Wald test for exogeneity of the endogeneous variable (to know whether it is necessary to run 2SLS)
#print(Three_vars_2SLS.wooldridge_regression)

########
# Runing the 2SLS with all control variables
Full_2SLS = IV2SLS(y, X, endog_var, Instrument).fit()
print(Full_2SLS.summary)
#print(Full_2SLS.first_stage)

# Wald test for exogeneity of the endogeneous variable (to know whether it is necessary to run 2SLS)
print(Full_2SLS.wooldridge_regression)


################
# Exporting the results of the regressions to Word
results_OLS = compare({"(1)": Two_vars_OLS, "(2)": Three_vars_OLS, "(3)": Full_OLS},
                  stars=True,
                  precision = "std_errors")

print(results_OLS)

# Converting the results to html
results_to_html = results_OLS.summary.as_html()
# Creating an html file where the data will be stored
file_path = 'OLS.html'
# Writing the file
with open(file_path, 'w', newline='', encoding='utf-8') as file:
    file.write(results_to_html)
# After creating the Html file, convert it to Microsoft Words
output = pypandoc.convert_file(file_path, 'docx', outputfile="OLS.docx")

################
# Exporting the results of the regressions to Word
results_2SLS= compare({"(1)": Two_vars_2SLS, "(2)": Three_vars_2SLS, "(3)": Full_2SLS},
                  stars=True,
                  precision = "std_errors")

print(results_2SLS)

# Converting the results to html
results_to_html = results_2SLS.summary.as_html()
# Creating an html file where the data will be stored
file_path = '2SLS.html'
# Writing the file
with open(file_path, 'w', newline='', encoding='utf-8') as file:
    file.write(results_to_html)
# After creating the Html file, convert it to Microsoft Words
output = pypandoc.convert_file(file_path, 'docx', outputfile="2SLS.docx")



##############################################################################
############################ 2nd Panel data analyses ########################
##############################################################################

# Replacing the values with the forcasted ones
df = df_copy

# Keeping the variables where there are some values in Malaysia
df = df_copy[['PM25 microgram per cubic metter','Province', 'Country name',
       'LC_urban_areas', 'Average_wind_speed', 'Pop density heads per Sqkm',
       'Average_precipitation', 'Maximum_temperature', 'Buses',
       'Motorcycles', 'Green_area']]

df = df.dropna()
df = df[df['Country name'] == 'Malaysia']


# Cleaning the dataset
# Adding a very small value to ensure the variable buses can transform to log
df['Log_Buses'] = np.log(df['Buses'] + 0.0001)
df['Log_PM25'] = np.log(df['PM25 microgram per cubic metter'])
df['Log_urban_areas'] = np.log(df['LC_urban_areas'])
df['Log_green_area'] = np.log(df['Green_area'])

# Summary statistics
Summary = df.describe(percentiles=[0.5]).T
print(Summary)
Summary.to_excel("Summary Statistics.xlsx")

# Histogram of the variables
histograms_of_data = df.hist(bins=30, figsize=(15, 10))
fig = histograms_of_data[0, 0].get_figure()
fig.savefig('Histograms_figure_5_countries.png')

## Setting the data for regression analyses / defining the variable
# Changing the variable type to ensure thy will considered as dummy in the regression
df = df.reset_index()
df['Id'] = df['Id'].astype('category')
df['year'] = df['year'].astype('category')

# Adding the constant variable
df = sm.add_constant(df)

# Defining the controls variables
X = df[['const','Average_wind_speed','Average_precipitation','Maximum_temperature',
         'Motorcycles','Log_green_area','Id']]

# Defining the variables of interest
y = df['Log_PM25']
endog_var = df['Log_Buses']
Instrument = df['Log_urban_areas']


######################### Running the regressions ##############################
########
## Performing an OLS (panel data fixed effect regression) to compare results with 2SLS
X_OLS = df[['const','Log_Buses','Average_wind_speed','Average_precipitation','Maximum_temperature',
        'Motorcycles', 'Log_green_area', 'Id']]

########
# Runing the OLS with no control variable
Two_vars_OLS = IV2SLS(y, df[['const', 'Log_Buses', 'Id']], None, None).fit()
#print(Two_vars_OLS.summary)

########
# Runing the OLS with greeneries as control variable
Three_vars_OLS= IV2SLS(y, df[['const', 'Log_green_area', 'Log_Buses', 'Id']], None, None).fit()
#print(Three_vars_OLS.summary)

########
# Runing the OLS with all control variables
Full_OLS = IV2SLS(y, X_OLS, None, None).fit()
#print(Full_OLS.summary)

################
# Runing the 2SLS with no control variable
Two_vars_2SLS = IV2SLS(y, df[['const', 'Id']], endog_var, Instrument).fit()
#print(Two_vars_2SLS.summary)
#print(Two_vars_2SLS.first_stage)
F1 = Two_vars_2SLS.first_stage

# Wald test for exogeneity of the endogeneous variable (to know whether it is necessary to run 2SLS)
#print(Two_vars_2SLS.wooldridge_regression)

########
# Runing the 2SLS with greeneries as control variable
Three_vars_2SLS = IV2SLS(y, df[['const', 'Id', 'Log_green_area']], endog_var, Instrument).fit()
#print(Three_vars_2SLS.summary)
#print(Three_vars_2SLS.first_stage)
F2 = Three_vars_2SLS.first_stage

# Wald test for exogeneity of the endogeneous variable (to know whether it is necessary to run 2SLS)
#print(Three_vars_2SLS.wooldridge_regression)

########
# Runing the 2SLS with all control variables
Full_2SLS = IV2SLS(y, X, endog_var, Instrument).fit()
#print(Full_2SLS.summary)


# Wald test for exogeneity of the endogeneous variable (to know whether it is necessary to run 2SLS)
#print(Full_2SLS.wooldridge_regression)


################
## Perfom the first stages of the 2sls regressions
First_stage_2_vars_2SLS = IV2SLS(endog_var, df[['const', 'Id', 'Log_urban_areas']], None, None).fit()
First_stage_3_vars_2SLS = IV2SLS(endog_var, df[['const', 'Id', 'Log_urban_areas', 'Log_green_area']], None, None).fit()
First_stage_Full_2SLS = IV2SLS(endog_var,df[['const','Average_wind_speed','Average_precipitation',
                                             'Maximum_temperature','Motorcycles', 'Log_urban_areas',
                                             'Log_green_area','Id']] , None, None).fit()

################
# Exporting the results of the regressions to Word
results_OLS = compare({"(1)": Two_vars_OLS, "(2)": Three_vars_OLS, "(3)": Full_OLS},
                  stars=True,
                  precision = "std_errors")


print(results_OLS)

# Converting the results to html
results_to_html = results_OLS.summary.as_html()
# Creating an html file where the data will be stored
file_path = 'OLS.html'
# Writing the file
with open(file_path, 'w', newline='', encoding='utf-8') as file:
    file.write(results_to_html)
# After creating the Html file, convert it to Microsoft Words
output = pypandoc.convert_file(file_path, 'docx', outputfile="OLS_5_countries.docx")

################
# Exporting the results of the regressions to Word
results_2SLS= compare({"(1)": Two_vars_2SLS, "(2)": Three_vars_2SLS, "(3)": Full_2SLS},
                  stars=True,
                  precision = "std_errors")

print(results_2SLS)

# Converting the results to html
results_to_html = results_2SLS.summary.as_html()
# Creating an html file where the data will be stored
file_path = '2SLS.html'
# Writing the file
with open(file_path, 'w', newline='', encoding='utf-8') as file:
    file.write(results_to_html)
# After creating the Html file, convert it to Microsoft Words
output = pypandoc.convert_file(file_path, 'docx', outputfile="2SLS_5_countries.docx")

################
# Exporting the results of the regressions to Word
results_1st_stage = compare({"(1)": First_stage_2_vars_2SLS,
                       "(2)": First_stage_3_vars_2SLS,
                       "(3)": First_stage_Full_2SLS},
                  stars=True,
                  precision = "std_errors")


print(results_1st_stage)

# Converting the results to html
results_to_html = results_1st_stage.summary.as_html()
# Creating an html file where the data will be stored
file_path = 'First_stage.html'
# Writing the file
with open(file_path, 'w', newline='', encoding='utf-8') as file:
    file.write(results_to_html)
# After creating the Html file, convert it to Microsoft Words
output = pypandoc.convert_file(file_path, 'docx', outputfile="First_stage_5_countries.docx")



###############################################################################
###############################################################################
###############################################################################
###############################################################################


from linearmodels.panel import PanelOLS, RandomEffects

# Replacing the values with the forcaseted ones
df_copy = pd.read_csv('Panel Data all countries.csv')
df = df_copy
df = df.dropna()

# Cleaning the dataset
# Adding a very small value to ensure the variable buses can transform to log
df['Log_Buses'] = np.log(df['Buses'] + 0.0001)
df['Log_PM25'] = np.log(df['PM25 microgram per cubic metter'])
df['Log_urban_areas'] = np.log(df['LC_urban_areas'])
df['Log_green_area'] = np.log(df['Green_area'])

## Testing for the best models of the Panel data
df['year'] = df['year'].astype('int')
df_panel = df.set_index(['Id', 'year'])

y_panel = df_panel['Log_PM25']
X_panel = df_panel[['Average_wind_speed','Average_precipitation','Maximum_temperature',
                    'Motorcycles', 'Log_urban_areas','Log_green_area']]
sm.add_constant(X_panel)

# Pooled OLS 
mod = PooledOLS(y_panel, X_panel)
pooled_res = mod.fit()
print(pooled_res)

# Fixed effects regression
mod = PanelOLS(y_panel, X_panel, entity_effects=True)
fe_res = mod.fit()
print(fe_res)

# Random effects regression
mod = RandomEffects(y_panel, X_panel)
re_res = mod.fit()
print(re_res)

# Coefficients of the demeaned values of the mean
## The closer the value is to 1, the closer the estimate of the random effect is to that of the fixed effects
## The closer the value is to 0, the closer the estimate of the random effect is to that of the pooled OLS estimate
re_res.variance_decomposition

# We use the hausman to determine is the data requires fixed effects estimations
def hausman(fe, re):
    ''' Estimation of the haussman test by hand. Basically it consists in 
    checking whether the coefficients of the fixed effects and the random
    effects are different.
    
    Use the estimate of the fixed effects first!
    
    H0: Both the random and the fixed effects are consistent with the data'''

    b = fe.params
    B = re.params
    v_b = fe.cov
    v_B = re.cov
    df = b[np.abs(b) < 1e8].size
    chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B))
    pval = stats.chi2.sf(chi2, df)
    
    
    return {
        "The value of the Chi-squared": chi2,
        "The degree of freedom": df,
        "The P-value": pval
    }

# Running the hausman test
hausman(fe_res, re_res)