# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:26:05 2024

@author: Coulibaly Yerema
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects
import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import pypandoc
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.linalg as la  
from scipy import stats
from linearmodels.iv import IV2SLS, compare
import numpy as np 
import os

# Showing all columns in the console
pd.set_option('display.max_columns', None)

# Changing the directory path
os.chdir("Path_to_files")

##############################################################################
########################### Working with Indonesia data ######################
##############################################################################

total_df = pd.read_csv('GDL_Atmospheric_data_5_ASEAN_countries.csv')

# Keeping Indonesian data from the atmospheric dataset
Indonesia_df = total_df[total_df['iso3'] == "IDN"] 

# Importing the vehicles data
df_nbr_vehicles = pd.read_csv("Number of Vehicles 2015-2022.csv")

# Replacing the names of Provinces that appear to be the same but are labelled differently across dataset
df_nbr_vehicles['Province'] = df_nbr_vehicles['Province'].replace('Kep. Riau','Kepulauan Riau')
df_nbr_vehicles['Province'] = df_nbr_vehicles['Province'].replace('DKI Jakarta','Jakarta Raya')
df_nbr_vehicles['Province'] = df_nbr_vehicles['Province'].replace('Kepulauan Bangka Belitung', 'Bangka Belitung')
df_nbr_vehicles['Province'] = df_nbr_vehicles['Province'].replace('DI Yogyakarta', 'Yogyakarta')

# Mering the data
df = pd.merge(Indonesia_df, df_nbr_vehicles, on=['Province','year'], how='outer')
df = df.sort_values(['Province','year']) # Sorting the values of the merged data to improve readability

# Saving th output for Indonesian data; filling the 'country name' column; and deleting the intermediary files
df_IDN = df
df_IDN['Country name'] = 'Indonesia'
del df_nbr_vehicles, Indonesia_df, df

##############################################################################
########################### Working with Phillipines data ####################
##############################################################################

# Importing vehicles data in Philippines
df_PHL_vehicles = pd.read_excel("Data of Philippines.xlsx", sheet_name="Sheet2")

# Renaming the column names to match the names of those of other datasets
df_PHL_vehicles = df_PHL_vehicles.rename(columns={'Long_name':'Province',
                                                  'Year':'year'})

# Keeping Phillipines data from the atmospheric dataset (total_df)
df_PHL = total_df[total_df['iso3'] == "PHL"]

# Replacing the names of Provinces that appear to be the same but are labelled differently across dataset
df_PHL['Province'] = df_PHL['Province'].replace('Cordillera Admin Region','Cordilera Administrative Region')
df_PHL['Province'] = df_PHL['Province'].replace('I-Ilocos','IIocos Region')
df_PHL['Province'] = df_PHL['Province'].replace('II-Cagayan Valley','Cagayan Valley')
df_PHL['Province'] = df_PHL['Province'].replace('III-Central Luzon','Central Luzon')
df_PHL['Province'] = df_PHL['Province'].replace('IVA-CALABARZON','Calabarzon')
df_PHL['Province'] = df_PHL['Province'].replace('IVB-MIMAROPA','Mimaropa')
df_PHL['Province'] = df_PHL['Province'].replace('V-Bicol','Bicol Region')
df_PHL['Province'] = df_PHL['Province'].replace('VI-Western Visayas','Western Visayas')
df_PHL['Province'] = df_PHL['Province'].replace('VII-Central Visayas','Central Visayas')
df_PHL['Province'] = df_PHL['Province'].replace('VIII-Eastern Visayas','Eastern Visayas')
df_PHL['Province'] = df_PHL['Province'].replace('IX-Zamboanga Peninsula','Zamboanga Peninsula')
df_PHL['Province'] = df_PHL['Province'].replace('XI-Davao','Davao Region')
df_PHL['Province'] = df_PHL['Province'].replace('XII-SOCCSKSARGEN','Soccsksargen')
df_PHL['Province'] = df_PHL['Province'].replace('XIII-Caraga','Caraga')
df_PHL['Province'] = df_PHL['Province'].replace('X-Northern Mindanao','North Mindanao')

# Merging the vehicles data with the atmospheric data (I am making the merge on a copy of the dataframe. if eveything is alright replace the original dataframe and drop the copy)
df_PHL_merged = pd.merge(df_PHL_vehicles, df_PHL, on=['Province', 'year'], how='outer')
df_PHL_merged = df_PHL_merged.sort_values(['Province','year']) # Sorting the values of the merged data to improve readability
df_PHL = df_PHL_merged # Replacing the copy with the original if everything is alright
df_PHL['Country name'] = 'Philippines'

# Dropping intermediary variables
del df_PHL_merged, df_PHL_vehicles


##############################################################################
########################### Working with Singapore data ######################
##############################################################################

# Importing vehicles data for Singapore
df_SGP_vehicles = pd.read_csv('Motor Vehicle Population By Type Of Vehicle 1990 - 2022.csv')

# Renaming the column names to match the names of those of other datasets
df_SGP_vehicles = df_SGP_vehicles.rename(columns={'Country':'Country name'})

# Keeping Singapore data from the atmospheric dataset (total_df)
df_SGP = total_df[total_df['iso3']=='SGP']
df_SGP['Province']='Singapore' # Renaming the content of the province column to Singapore as there is only one province

# Merging the vehicles data with the atmospheric data (I am making the merge on a copy of the dataframe. if eveything is alright replace the original dataframe and drop the copy)
df_SGP_merged = pd.merge(df_SGP, df_SGP_vehicles, on=['Country name', 'year'], how='outer') # For the special case of Singapore the merging is done at the country level because all data are at the country level
df_SGP_merged = df_SGP_merged.sort_values(['Country name','year'])  # Sorting the values of the merged data to improve readability
df_SGP = df_SGP_merged # Replacing the copy with the original if everything is alright

# Dropping intermediary variables
del df_SGP_merged, df_SGP_vehicles

##############################################################################
########################### Working with Malaysia data #######################
##############################################################################

# Importing vehicles data in Philippines
df_MYS_vehicles = pd.read_excel("MalaysiaData.xlsx", header=1)
print(df_MYS_vehicles.columns)
df_MYS_vehicles = df_MYS_vehicles[['Year', 'State','Motorcycle', 'Public transport']]

# Renaming the column names to match the names of those of other datasets
df_MYS_vehicles = df_MYS_vehicles.rename(columns={'Year':'year',
                                                  'State':'Province',
                                                  'Motorcycle':'Motorcycles',
                                                  'Public transport': 'Buses'})

# Keeping Malaysia data from the atmospheric dataset (total_df)
df_MYS = total_df[total_df['iso3'] == "MYS"]

# Checking if there is a discrepancy in name across dataset
list_vehicles = df_MYS_vehicles.Province.unique() # Extracting the unique values of provinces in the vehicle dataset
list_vehicles.sort() # Sorting the unique province names to ease comparisons
list_atmospheric = df_MYS.Province.unique() # Extracting the unique values of provinces in the atmospheric dataset
list_atmospheric.sort() # Sorting the unique province names to ease comparisons
print(list_vehicles) # Printing the lists
print(list_atmospheric) # Printing the lists

# Replacing the names of Provinces that appear to be the same but are labelled differently across dataset
df_MYS['Province'] = df_MYS['Province'].replace('Melaka','Malacca')
df_MYS['Province'] = df_MYS['Province'].replace('Pulau Pinang','Penang')
df_MYS['Province'] = df_MYS['Province'].replace('Trengganu','Terengganu')

# Merging the vehicles data with the atmospheric data (I am making the merge on a copy of the dataframe. if eveything is alright replace the original dataframe and drop the copy)
df_MYS_merged = pd.merge(df_MYS_vehicles, df_MYS, on=['Province', 'year'], how='outer')
df_MYS_merged = df_MYS_merged.sort_values(['Province', 'year']) # Sorting the values of the merged data to improve readability
df_MYS = df_MYS_merged # Replacing the copy with the original if everything is alright
df_MYS['Country name'] = 'Malaysia'

# Dropping intermediary variables
del df_MYS_merged, df_MYS_vehicles, list_atmospheric, list_vehicles


##############################################################################
########################### Working with Thaialnd data #######################
##############################################################################

# Importing vehicles data in Philippines
df_THA_vehicles = pd.read_excel("Thailand (Southern region).xlsx")
#print(df_THA_vehicles.columns)
df_THA_vehicles = df_THA_vehicles.rename(columns = {'Year': 'year'})

df_THA_vehicles2 = pd.read_excel("Thailand Data.xlsx", sheet_name='Sheet3')
#print(df_THA_vehicles2.columns)
df_THA_vehicles2 = df_THA_vehicles2.rename(columns={'Car for hire(Sedan)':'Car for hire (Sedan)',
                                                    'Microbus & amp; passenger pick up':'Microbus &amp; passenger pick up',
                                                    'Hotel taxi(Sedan)':'Hotel taxi (Sedan)',
                                                    })

df_THA_vehicles = pd.concat([df_THA_vehicles, df_THA_vehicles2])
print(df_THA_vehicles.columns)

# Creating the vehicles data
df_THA_vehicles['Cars'] = df_THA_vehicles[['Car for hire (Sedan)',
                                           'Hotel taxi (Sedan)',
                                           'Sedan (Not more than 7 passengers)',
                                           'Tour taxi (Sedan)',
                                           'Urban taxi (Sedan)'
                                           ]].sum(axis=1)
df_THA_vehicles['Buses'] = df_THA_vehicles[['Fixed route bus',
                                           'Non-fixed route bus',
                                           'Small rural bus',
                                           'Private bus'
                                           ]].sum(axis=1)
df_THA_vehicles['Trucks'] =df_THA_vehicles[['Non-fixed route truck',
                                           'Non-fixed route bus',
                                           'Private truck',
                                           'Van & Pick Up',
                                           ]].sum(axis=1)
df_THA_vehicles['Motorcycles'] = df_THA_vehicles[['Motorcycle',
                                           'Motortricycle',
                                           'Public Motorcycle'
                                           ]].sum(axis=1)

df_THA_vehicles = df_THA_vehicles[['year', 'Province','Motorcycles', 'Buses',
                                   'Cars', 'Trucks']]

# Loop through each province in the DataFrame
# =============================================================================
# for province in df_THA_vehicles['Province'].unique():
#     # Filter the DataFrame for the current province
#     df_province = df_THA_vehicles[df_THA_vehicles['Province'] == province]
#     
#     # Create a scatter plot for the current province
#     plt.figure(figsize=(10, 6))  # Set the figure size for better readability
#     plt.scatter(df_province['year'], df_province['Buses'], label='Buses')
#     
#     # Adding labels and title
#     plt.xlabel('Year')
#     plt.ylabel('Number of Buses')
#     plt.title(f'Scatter Plot of Buses Over Years in {province}')
#     plt.legend()
#     
#     # Show the plot
#     plt.show()
# =============================================================================


# Keeping Thailand data from the atmospheric dataset (total_df)
df_THA = total_df[total_df['iso3'] == "THA"]

# Checking if there is a discrepancy in name across dataset
list_vehicles = df_THA_vehicles.Province.unique() # Extracting the unique values of provinces in the vehicle dataset
list_vehicles.sort() # Sorting the unique province names to ease comparisons
list_atmospheric = df_THA.Province.unique() # Extracting the unique values of provinces in the atmospheric dataset
list_atmospheric.sort() # Sorting the unique province names to ease comparisons
print(list_vehicles) # Printing the lists
print(list_atmospheric) # Printing the lists


df_THA_vehicles.loc[df_THA_vehicles['Province']=='Bangkok']



# Merging the vehicles data with the atmospheric data (I am making the merge on a copy of the dataframe. if eveything is alright replace the original dataframe and drop the copy)
df_THA_merged = pd.merge(df_THA_vehicles, df_THA, on=['Province', 'year'], how='outer')
df_THA_merged = df_THA_merged.sort_values(['Province', 'year']) # Sorting the values of the merged data to improve readability
df_THA = df_THA_merged # Replacing the copy with the original if everything is alright
df_THA['Country name'] = 'Thailand'

# Dropping intermediary variables
del df_THA_merged, df_THA_vehicles, list_atmospheric, list_vehicles

##############################################################################
########################### Appending the data across countries ##############
##############################################################################

# Appending the data
df = pd.concat([df_IDN, df_MYS, df_PHL, df_SGP, df_THA])

# Sorting the values
df.sort_values(['Country name','Province','year'])


# Creating a green area variable
df['Green_area'] = df['LC_irrigated_cropland'] + df['LC_mosaic_cropland'] + df['LC_natural_Vegetation']+ df['LC_tree_broadleaved'] + df['VALUE_70'] + df['LC_tree_needleaved'] + df['LC_tree_shrub'] + df['LC_herbacious'] + df['LC_grassland'] + df['LC_sparce_vegetation'] + df['LC_cropland'] + df['LC_shrubland'] + df['LC_tree_flooded']


# Sliced
df = df[['PM25 microgram per cubic metter', 'year', 'Province',
         'Country name','LC_urban_areas', 'Average_wind_speed',
         'Pop density heads per Sqkm', 'Average_precipitation',
         'Maximum_temperature', 'Cars', 'Buses', 'Trucks', 'Motorcycles',
         'Green_area']]

# Dropping observations for years that we will not use
df = df[(df['year'] <=2022) & (df['year'] >= 2012)] 

# Dropping observations where we do not have the dependent variable
df = df.dropna(subset=['PM25 microgram per cubic metter'])
df = df[df['Province'] != 'Labuan'] # I drop observations from Labuan because the island is small that ArcGIS fails to make reliable estiamte of the atmospheric data 

# Checking the data
df.describe().T # Summary statistics: It appears that 'trucks' is not recognised as a numeric value
df['Trucks'] = df['Trucks'].astype(float)

##############################################################################
#### Times Series analyses to fill missing values of independent variables ###
##############################################################################

# Setting a unique Id for all set of province and country
df['dummy'] = df['Country name'] + df['Province']
df['Id'] = pd.factorize(df['dummy'])[0]
del df['dummy']

# Set as panel data
df = df.set_index(["Id", "year"]) # Setting hierarchical indices to facilitate interpolation
df = df.sort_index() # Sorting the index as good practice

df = df.sort_index() 

# Creating a copy of the data for Time Series analyses
df_copy = df

# Selection of index
first_level = df_copy.index.get_level_values(0).unique()

# Extrapolating the data with time series analyses
# Selection of variables to be extrapolated
variables = ["LC_urban_areas","Pop density heads per Sqkm", "Cars", "Buses",
             "Trucks", "Motorcycles","Green_area"]

Times_series_df = pd.DataFrame(columns=['Variable name', 'Province', 'Country name','Parameters'])

# Selection of index
first_level = df_copy.index.get_level_values(0).unique()
# Looping over the variables whose values we want to extrapolate
for var in variables:
    print(var)
    
    # Selecting all rows where the first-level index meets a condition
    mask = df_copy.index.get_level_values(0) == 0
    selected_data = df_copy[mask]
    
    if not selected_data[var].isna().all():
        # Transforming the index of the data to get fit a time series regression
        selected_data = selected_data.reset_index()
        selected_data = selected_data.set_index('year')
        selected_data.index = selected_data.index.astype(int)
        selected_data.index = selected_data.index.astype(str)
        selected_data.index = pd.to_datetime(selected_data.index, format='%Y')
        #selected_data = selected_data.asfreq('AS')
            
        # Attempting to find the parameters p, q and d for fitting the ARIMA
        # Estimating the unit root of the time series (d)
        d = 0
        try:
            ADFtest = adfuller(selected_data[var].dropna(),
                               maxlag=None,
                               regression='ct',
                               autolag='AIC',
                               store=False,
                               regresults=False)
        
            while (ADFtest[1] > 0.1) & (d <= 2): # We use 0.1 as critical value and we stop the differencing after 2 turns because we do not have much data to go beyond that.  
                d = d + 1 
                dif = selected_data[var].diff().dropna()
                
                ADFtest = adfuller(dif.dropna(),
                                   maxlag=None,
                                   regression='ct',
                                   autolag='AIC',
                                   store=False,
                                   regresults=False)
        except:
            print("ADF failed")
            pass
        
        lag_errors, lags_dependent = 1, 1
        try:
            # Performing the auto-corelation and extracting the statistics from it to find the value q.
            acf_values = acf(selected_data[var].dropna(), alpha=0.1) # We use 0.1 as critical value
            
            # Extracting the number of lags that may be significant  for the parameter q
            acf_values_q_conf_interval = acf_values[1] # Extracting the confidence interval of the correlation of each lag with contemporary values.
            lag_errors = np.prod(acf_values_q_conf_interval, axis=1) # Multiplying the the high bound and the low bound of the confidence interval. The quotien of this multiplication should yield nnegative values for non-significant results and positive values fo significant results.
            lag_errors = len(lag_errors[lag_errors>0]) # Counting the number of lags with significant values.
        
        except:
            print("Auto-correlation failed")
            pass
        
        try:
            # Performing the partial auto-corelation and extracting the statistics from it to find the value p.
            pacf_values = pacf(selected_data[var].dropna(), nlags=2, alpha=0.1) # We use 0.1 as critical value
        
            # Extracting the number of lags that may be significant for the parameter p      
            acf_values_p_conf_interval = pacf_values[1] # Extracting the confidence interval of the correlation of each lag with contemporary values.
            lags_dependent = np.prod(acf_values_p_conf_interval, axis=1)  # Multiplying the the high bound and the low bound of the confidence interval. The quotien of this multiplication should yield nnegative values for non-significant results and positive values fo significant results
            lags_dependent = len(lags_dependent[lags_dependent>0]) # Counting the number of lags with significant values
            
        except:
            print("Partial auto-correlation failed")
            pass
        
        # Times Series regression with the different combinasons of the paramters saved during the previous steps
        models = {}
        for lags_1 in range(0,lags_dependent):
            for lags_2 in range(0,lag_errors):
                try:
                    model = ARIMA(selected_data[var], order=(lags_1,d,lags_2), trend ='ct')
                except:
                    d = 0
                    model = ARIMA(selected_data[var], order=(lags_1,d,lags_2), trend ='ct')
                    
                model_fit = model.fit()                
                models[model_fit.aic] = model # Storing the values of the models and AIC
        
        # Selecting the best ARIMA models                  
        lowest_AIC = models.keys() # Selecting the AIC from dictionnary
        lowest_AIC = sorted(lowest_AIC)[0] # Sorting all the AIC values collected and selecting the values that is the lowest
        best_model = models[lowest_AIC] # Retriving the model with the lowest AIC
        best_model_fit = best_model.fit() # Storing the best model
    
        # Predicting values based on the times series estimates
        forecast = best_model_fit.predict(start=selected_data.index[0], end=selected_data.index[-1])
        forecast[forecast < 0] = 0
        
        # Checking the results visually
        # Plotting
# =============================================================================
#         plt.figure(figsize=(10, 6))  
#         plt.plot(selected_data[var], label='Actual '+var, marker='o')  
#         plt.plot(forecast, label='Forecasted '+var, marker='x')  
#         plt.title('Actual vs Forecasted '+var)
#         plt.xlabel('Year')  
#         plt.ylabel('Number of '+var)
#         plt.legend()  
#         plt.xticks(rotation=45) 
#         plt.grid(True)
#         plt.show()
# =============================================================================
        
        # Replacing the values of missing variables 
        selected_data[var] = selected_data[var].fillna(forecast)
        
        # Reseting the indices to the original data
        selected_data = selected_data.reset_index()
        selected_data['year'] = selected_data['year'].dt.year
        selected_data['year'] = selected_data['year'].astype(float)
        selected_data = selected_data.set_index(["Id", "year"])
   
    else:
        print("outside if statement")
        pass

    selected_data_original = selected_data

    Province = selected_data['Province'].unique()[0]
    Country = selected_data['Country name'].unique()[0]
    new_data = pd.DataFrame([{'Variable name': var, 'Province': Province, 'Country name' : Country,'Parameters': best_model.order}])
    Times_series_df = pd.concat([Times_series_df, new_data], ignore_index=True)
    
    for province in first_level[1:]:
        # Select all rows where the first-level index meets a condition
        mask = df_copy.index.get_level_values(0) == province
        selected_data = df_copy[mask]
        
        if not selected_data[var].isna().all():
        
            # Transforming the index of the data to get fit a time series regression
            selected_data = selected_data.reset_index()
            selected_data = selected_data.set_index('year')
            selected_data.index = selected_data.index.astype(int)
            selected_data.index = selected_data.index.astype(str)
            selected_data.index = pd.to_datetime(selected_data.index, format='%Y')
            #selected_data = selected_data.asfreq('AS')
            
            # Ensuring we work only with data that have at least one sample point
            try:
                # Attempting to find the parameters p, q and d for fitting the ARIMA
                # Estimating the unit root of the time series (d)
                d = 0
                ADFtest = adfuller(selected_data[var].dropna(),
                                   maxlag=None,
                                   regression='ct',
                                   autolag='AIC',
                                   store=False,
                                   regresults=False)
        
                while (ADFtest[1] > 0.1) & (d <= 2): # We use 0.1 as critical value and we stop the differencing after 2 turns because we do not have much data to go beyond that.
                    d = d + 1 
                    dif = selected_data[var].diff().dropna()
                    
                    ADFtest = adfuller(dif.dropna(),
                                       maxlag=None,
                                       regression='ct',
                                       autolag='AIC',
                                       store=False,
                                       regresults=False)
            except:
                print("ADF failed")
                pass
        
            lag_errors, lags_dependent = 1, 1
            
            try:
                # Performing the auto-corelation and extracting the statistics from it to find the value q.
                acf_values = acf(selected_data[var].dropna(), alpha=0.1) # We use 0.1 as critical value
                
                # Extracting the number of lags that may be significant  for the parameter q
                acf_values_q_conf_interval = acf_values[1] # Extracting the confidence interval of the correlation of each lag with contemporary values
                lag_errors = np.prod(acf_values_q_conf_interval, axis=1) # Multiplying the the high bound and the low bound of the confidence interval. The quotien of this multiplication should yield nnegative values for non-significant results and positive values fo significant results
                lag_errors = len(lag_errors[lag_errors>0]) # Counting the number of lags with significant values.
            
            except:
                print("Auto-correlation failed")
                pass
            
            try:
                # Performing the partial auto-corelation and extracting the statistics from it to find the value p.
                pacf_values = pacf(selected_data[var].dropna(), nlags=2, alpha=0.1) # We use 0.1 as critical value
        
                # Extracting the number of lags that may be significant for the parameter p      
                acf_values_p_conf_interval = pacf_values[1] # Extracting the confidence interval of the correlation of each lag with contemporary values
                lags_dependent = np.prod(acf_values_p_conf_interval, axis=1)  # Multiplying the the high bound and the low bound of the confidence interval. The quotien of this multiplication should yield nnegative values for non-significant results and positive values fo significant results
                lags_dependent = len(lags_dependent[lags_dependent>0]) # Counting the number of lags with significant values
            
            except:
                print("Partial auto-correlation failed")
                pass
            
            # Times Series regression with the different combinasons of the paramters saved during the previous steps
            models = {}
            for lags_1 in range(0,lags_dependent):
                for lags_2 in range(0,lag_errors):
                    try:
                        model = ARIMA(selected_data[var], order=(lags_1,d,lags_2), trend ='ct')
                    except:
                        model = ARIMA(selected_data[var], order=(lags_1,0,lags_2), trend ='ct')
                    
                    model_fit = model.fit()
                    models[model_fit.aic] = model # Storing the values of the models and AIC
            
            # Selecting the best ARIMA models                  
            lowest_AIC = models.keys()
            lowest_AIC = sorted(lowest_AIC)[0]
            best_model = models[lowest_AIC]
            best_model_fit = best_model.fit()
    
            # Predicting values based on the times series estimates
            forecast = best_model_fit.predict(start=selected_data.index[0], end=selected_data.index[-1])
            forecast[forecast < 0] = 0
            
            # Checking the results visually
            # Plotting
# =============================================================================
#             plt.figure(figsize=(10, 6))  
#             plt.plot(selected_data[var], label='Actual '+var, marker='o')  
#             plt.plot(forecast, label='Forecasted '+var, marker='x')  
#             plt.title('Actual vs Forecasted '+var)
#             plt.xlabel('Year')  
#             plt.ylabel('Number of '+var) 
#             plt.legend()  
#             plt.xticks(rotation=45) 
#             plt.grid(True)
#             plt.show()
# =============================================================================
            
            # Replacing the values of missing variables 
            selected_data[var] = selected_data[var].fillna(forecast)
            
            # Reseting the indices to the original data
            selected_data = selected_data.reset_index()
            selected_data['year'] = selected_data['year'].dt.year
            selected_data['year'] = selected_data['year'].astype(float)
            selected_data = selected_data.set_index(["Id", "year"])

        else:
            print("outside if statement")
            pass

        # Append
        selected_data_original = pd.concat([selected_data_original, selected_data])
        
        Province = selected_data['Province'].unique()[0]
        Country = selected_data['Country name'].unique()[0]
        new_data = pd.DataFrame([{'Variable name': var, 'Province': Province, 'Country name' : Country,'Parameters': best_model.order}])
        Times_series_df = pd.concat([Times_series_df, new_data], ignore_index=True)
        
    # Storing the final dataset
    df_copy = selected_data_original

# Deleting the intermediary variables
del acf_values, acf_values_p_conf_interval, acf_values_q_conf_interval, ADFtest, best_model, best_model_fit, d, dif, mask, selected_data, selected_data_original, var, lowest_AIC, lags_1, lags_2, lag_errors, lags_dependent, model_fit, model, models, pacf_values, first_level, forecast, province

df_copy.to_csv('../Panel Data all countries.csv')

Times_series_df['Area'] = Times_series_df['Province'] + " (" + Times_series_df['Country name'] + ")"
del Times_series_df['Province'], Times_series_df['Country name']

df_Times_Series = pd.DataFrame()
for var in variables:
    temporary = Times_series_df[Times_series_df['Variable name']==var]
    del temporary['Variable name']
    collapsed_df = temporary.groupby('Parameters').agg(lambda x: ', '.join(x.astype(str))).reset_index()
    collapsed_df['Number of provinces'] = collapsed_df['Area'].str.count(', ') + 1
    collapsed_df['Variable'] = var
    df_Times_Series = pd.concat([df_Times_Series, collapsed_df])
    

df_Times_Series.to_excel('Times Series per provinces.xlsx')
