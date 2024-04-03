# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import warnings
warnings.filterwarnings('ignore')

# +
import pandas as pd

# Load the data files
google_mobility_data = pd.read_csv('data/google_activity_by_London_Borough.csv')
covid_deaths_data = pd.read_csv('data/phe_deaths_london_boroughs.csv')
merged_data = pd.read_csv('data/merged_data.csv')

# Getting basic information about the datasets
google_mobility_info = google_mobility_data.info()
covid_deaths_info = covid_deaths_data.info()
merged_data_info = merged_data.info()

(google_mobility_info, covid_deaths_info, merged_data_info)


# +
import pandas as pd

# Assuming google_mobility_data, covid_deaths_data, and merged_data are pre-defined DataFrames

# Google Mobility Data Cleaning
google_mobility_data_cleaned = google_mobility_data.drop(columns=['Unnamed: 0'])
google_mobility_data_cleaned['date'] = pd.to_datetime(google_mobility_data_cleaned['date'])
for column in google_mobility_data_cleaned.select_dtypes(include=['float64', 'int64']).columns:
    google_mobility_data_cleaned[column].fillna(google_mobility_data_cleaned[column].mean(), inplace=True)

# COVID-19 Deaths Data Cleaning
covid_deaths_data_cleaned = covid_deaths_data.copy()
covid_deaths_data_cleaned['date'] = pd.to_datetime(covid_deaths_data_cleaned['date'])

# Merged Data Cleaning
merged_data_cleaned = merged_data.copy()
merged_data_cleaned['date'] = pd.to_datetime(merged_data_cleaned['date'])
for column in merged_data_cleaned.select_dtypes(include=['float64', 'int64']).columns:
    merged_data_cleaned[column].fillna(merged_data_cleaned[column].mean(), inplace=True)

# Descriptive Analysis
google_mobility_descriptive_stats = google_mobility_data_cleaned.describe()
covid_deaths_descriptive_stats = covid_deaths_data_cleaned.describe()
merged_data_descriptive_stats = merged_data_cleaned.describe()

# Output the descriptive statistics
google_mobility_descriptive_stats, covid_deaths_descriptive_stats, merged_data_descriptive_stats

# +
# Descriptive Analysis for the datasets

# Descriptive statistics for Google Mobility Data
google_mobility_descriptive_stats = google_mobility_data_cleaned.describe()

# Descriptive statistics for COVID-19 Deaths Data
covid_deaths_descriptive_stats = covid_deaths_data_cleaned.describe()

# Descriptive statistics for Merged Data
merged_data_descriptive_stats = merged_data_cleaned.describe()

google_mobility_descriptive_stats, covid_deaths_descriptive_stats, merged_data_descriptive_stats



# +
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Exploratory Data Analysis (EDA) with Visualization

# Google Mobility Data: Trends over Time
plt.figure(figsize=(15, 10))
plt.subplot(3, 2, 1)
sns.lineplot(data=google_mobility_data_cleaned, x='date', y='retail_and_recreation_percent_change_from_baseline', label='Retail & Recreation')
plt.title('Retail & Recreation Mobility Over Time')

plt.subplot(3, 2, 2)
sns.lineplot(data=google_mobility_data_cleaned, x='date', y='grocery_and_pharmacy_percent_change_from_baseline', label='Grocery & Pharmacy')
plt.title('Grocery & Pharmacy Mobility Over Time')

plt.subplot(3, 2, 3)
sns.lineplot(data=google_mobility_data_cleaned, x='date', y='parks_percent_change_from_baseline', label='Parks')
plt.title('Parks Mobility Over Time')

plt.subplot(3, 2, 4)
sns.lineplot(data=google_mobility_data_cleaned, x='date', y='transit_stations_percent_change_from_baseline', label='Transit Stations')
plt.title('Transit Stations Mobility Over Time')

plt.subplot(3, 2, 5)
sns.lineplot(data=google_mobility_data_cleaned, x='date', y='workplaces_percent_change_from_baseline', label='Workplaces')
plt.title('Workplaces Mobility Over Time')

plt.subplot(3, 2, 6)
sns.lineplot(data=google_mobility_data_cleaned, x='date', y='residential_percent_change_from_baseline', label='Residential')
plt.title('Residential Mobility Over Time')

plt.tight_layout()
plt.show()

# COVID-19 Deaths Data: Trends over Time
plt.figure(figsize=(10, 5))
sns.lineplot(data=covid_deaths_data_cleaned, x='date', y='new_deaths', label='New Deaths')
plt.title('COVID-19 New Deaths Over Time')
plt.show()



# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming google_mobility_data_cleaned and covid_deaths_data_cleaned are pre-defined DataFrames

# Selecting only numeric columns for mean calculation in Google Mobility Data
numeric_columns = google_mobility_data_cleaned.select_dtypes(include=['float64', 'int64']).columns
google_mobility_sampled = google_mobility_data_cleaned.groupby('area_name').resample('W', on='date')[numeric_columns].mean().reset_index()

# Plotting trends over time for Google Mobility Data
plt.figure(figsize=(15, 10))

# Define a function to create line plots
def create_line_plot(position, data, x, y, title):
    plt.subplot(3, 2, position)
    sns.lineplot(data=data, x=x, y=y, ci=None)
    plt.title(title)

# Create line plots for different categories
create_line_plot(1, google_mobility_sampled, 'date', 'retail_and_recreation_percent_change_from_baseline', 'Weekly Avg: Retail & Recreation')
create_line_plot(2, google_mobility_sampled, 'date', 'grocery_and_pharmacy_percent_change_from_baseline', 'Weekly Avg: Grocery & Pharmacy')
create_line_plot(3, google_mobility_sampled, 'date', 'parks_percent_change_from_baseline', 'Weekly Avg: Parks')
create_line_plot(4, google_mobility_sampled, 'date', 'transit_stations_percent_change_from_baseline', 'Weekly Avg: Transit Stations')
create_line_plot(5, google_mobility_sampled, 'date', 'workplaces_percent_change_from_baseline', 'Weekly Avg: Workplaces')
create_line_plot(6, google_mobility_sampled, 'date', 'residential_percent_change_from_baseline', 'Weekly Avg: Residential')

plt.tight_layout()
plt.show()

# Ensuring 'area_name' is set as an index
covid_deaths_data_cleaned.set_index('area_name', append=True, inplace=True)

# Group by 'area_name', resample, and sum the values
covid_deaths_weekly = covid_deaths_data_cleaned.groupby(level='area_name').resample('W', on='date').sum()

# Resetting the index
covid_deaths_weekly.reset_index(inplace=True)

plt.figure(figsize=(10, 5))
sns.lineplot(data=covid_deaths_weekly, x='date', y='new_deaths', ci=None)
plt.title('Weekly Sum: COVID-19 New Deaths')
plt.show()


# +
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# For a more efficient analysis, focusing on 'Retail and Recreation' mobility and COVID-19 'new_deaths'
# Aggregating data to weekly level for London overall
weekly_mobility = google_mobility_data_cleaned.groupby(pd.Grouper(key='date', freq='W'))['retail_and_recreation_percent_change_from_baseline'].mean()
weekly_deaths = covid_deaths_data_cleaned.groupby(pd.Grouper(key='date', freq='W'))['new_deaths'].sum()

# Time Series Decomposition for 'Retail and Recreation' mobility
decomposition_mobility = seasonal_decompose(weekly_mobility.dropna(), model='additive')
decomposition_deaths = seasonal_decompose(weekly_deaths.dropna(), model='additive')

# Plotting the decomposed time series of 'Retail and Recreation' mobility
plt.figure(figsize=(14, 8))
plt.subplot(4, 1, 1)
plt.plot(decomposition_mobility.observed)
plt.title('Retail and Recreation Mobility: Observed')
plt.subplot(4, 1, 2)
plt.plot(decomposition_mobility.trend)
plt.title('Trend')
plt.subplot(4, 1, 3)
plt.plot(decomposition_mobility.seasonal)
plt.title('Seasonal')
plt.subplot(4, 1, 4)
plt.plot(decomposition_mobility.resid)
plt.title('Residual')
plt.tight_layout()

plt.figure(figsize=(14, 8))
plt.subplot(4, 1, 1)
plt.plot(decomposition_deaths.observed)
plt.title('COVID-19 New Deaths: Observed')
plt.subplot(4, 1, 2)
plt.plot(decomposition_deaths.trend)
plt.title('Trend')
plt.subplot(4, 1, 3)
plt.plot(decomposition_deaths.seasonal)
plt.title('Seasonal')
plt.subplot(4, 1, 4)
plt.plot(decomposition_deaths.resid)
plt.title('Residual')
plt.tight_layout()

plt.show()

# Performing Augmented Dickey-Fuller test to check stationarity
adf_test_mobility = adfuller(weekly_mobility.dropna())
adf_test_deaths = adfuller(weekly_deaths.dropna())

adf_results = {
    "Retail and Recreation Mobility": adf_test_mobility[1],
    "COVID-19 New Deaths": adf_test_deaths[1]
}

adf_results

# +
# Advanced Correlation Analysis between 'Retail and Recreation' mobility and COVID-19 'new_deaths'

# Preparing the data for correlation analysis
# Ensuring both series have the same length and index
common_index = weekly_mobility.dropna().index.intersection(weekly_deaths.dropna().index)
mobility_aligned = weekly_mobility[common_index]
deaths_aligned = weekly_deaths[common_index]

# Advanced correlation analysis using Cross-Correlation
cross_correlation = sm.tsa.stattools.ccf(mobility_aligned, deaths_aligned, adjusted=False)

# Plotting the Cross-Correlation Function
plt.figure(figsize=(10, 6))
plt.stem(range(len(cross_correlation)), cross_correlation, use_line_collection=True)
plt.title('Cross-Correlation between Retail & Recreation Mobility and COVID-19 New Deaths')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')
plt.show()

# Identifying the lag at which correlation is strongest
max_correlation_lag = cross_correlation.argmax()
max_correlation_value = cross_correlation[max_correlation_lag]

max_correlation_lag, max_correlation_value



# +
from statsmodels.tsa.stattools import grangercausalitytests

# 1. Investigating Potential Causal Relationships using Granger Causality Tests
# Granger Causality Test requires a DataFrame with both time series
granger_df = pd.concat([mobility_aligned, deaths_aligned], axis=1)
granger_df.columns = ['Retail_Recreation_Mobility', 'COVID_Deaths']

# Performing Granger Causality Test with maxlag determined from previous analysis
max_lag = 5  # Using a smaller lag for a more practical analysis
granger_test_results = grangercausalitytests(granger_df, maxlag=max_lag, verbose=True)

# 2. Exploring Correlations at Different Phases of the Pandemic
# Identifying different pandemic phases based on government policies, restrictions, etc.
# Due to the complexity, here we only outline the approach:
# - Segment the data based on identified pandemic phases.
# - Perform correlation analysis for each phase separately.
# - Compare correlations across different phases to understand the impact of policies and behaviors.

# 3. Incorporating Additional Variables
# Outline of the approach:
# - Collect additional data on government restrictions, vaccination rates, and public health measures.
# - Integrate these variables into the analysis.
# - Perform multivariate correlation or regression analysis to understand the combined impact on mobility and COVID-19 deaths.

# Note: The actual execution of steps 2 and 3 requires additional data and is beyond the scope of this current analysis.
# The results and insights from these steps would depend on the specific data and methodology used.


# +
# Segmenting data based on provided policy phases for correlation analysis

# Defining the time periods for each pandemic phase
pandemic_phases = {
    "Phase 1": {"start": "2020-01-01", "end": "2020-03-23"},
    "Phase 2": {"start": "2020-03-24", "end": "2020-07-07"},
    "Phase 3": {"start": "2020-07-08", "end": "2020-10-30"},
    "Phase 4": {"start": "2020-10-31", "end": "2020-12-01"},
    "Phase 5": {"start": "2020-12-02", "end": "2020-12-31"},
    "Phase 6": {"start": "2021-01-01", "end": "2021-12-15"},
    "Phase 7": {"start": "2021-12-16", "end": "2021-12-31"},
    "Phase 8": {"start": "2022-01-01", "end": "2022-01-18"},
    "Phase 9": {"start": "2022-01-19", "end": "2022-03-17"},
    "Phase 10": {"start": "2022-03-18", "end": "2022-12-31"}  # Assuming end of 2022 for analysis
}

# Function to segment data based on phases
def segment_data(data, phases):
    segmented_data = {}
    for phase, duration in phases.items():
        mask = (data.index >= duration['start']) & (data.index <= duration['end'])
        segmented_data[phase] = data[mask]
    return segmented_data

# Segmenting the mobility and death data
segmented_mobility = segment_data(mobility_aligned, pandemic_phases)
segmented_deaths = segment_data(deaths_aligned, pandemic_phases)

# Calculating correlations for each phase
phase_correlations = {phase: segmented_mobility[phase].corr(segmented_deaths[phase]) for phase in pandemic_phases}

phase_correlations



# +
from statsmodels.formula.api import ols

# Multivariate Analysis Using Merged Data

# Selecting relevant variables for the multivariate analysis
# For simplicity, we will consider retail and recreation mobility, grocery and pharmacy mobility, 
# parks mobility, transit stations, workplaces, residential areas, and COVID-19 new deaths.
selected_columns = [
    'retail_and_recreation_percent_change_from_baseline',
    'grocery_and_pharmacy_percent_change_from_baseline',
    'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline',
    'new_deaths'
]

# Preparing the data for analysis
multivariate_data = merged_data_cleaned[selected_columns].dropna()

# Building the OLS model for multivariate regression
formula = 'new_deaths ~ retail_and_recreation_percent_change_from_baseline + grocery_and_pharmacy_percent_change_from_baseline + parks_percent_change_from_baseline + transit_stations_percent_change_from_baseline + workplaces_percent_change_from_baseline + residential_percent_change_from_baseline'
model = ols(formula, data=multivariate_data).fit()

# Summarizing the model results
model_summary = model.summary()
model_summary



# +
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Feature Engineering: Including Interaction Terms and Checking for Multicollinearity

# Adding interaction terms
multivariate_data['interaction_retail_grocery'] = multivariate_data['retail_and_recreation_percent_change_from_baseline'] * multivariate_data['grocery_and_pharmacy_percent_change_from_baseline']
multivariate_data['interaction_parks_transit'] = multivariate_data['parks_percent_change_from_baseline'] * multivariate_data['transit_stations_percent_change_from_baseline']

# Building the OLS model with interaction terms
formula_interactions = 'new_deaths ~ retail_and_recreation_percent_change_from_baseline + grocery_and_pharmacy_percent_change_from_baseline + parks_percent_change_from_baseline + transit_stations_percent_change_from_baseline + workplaces_percent_change_from_baseline + residential_percent_change_from_baseline + interaction_retail_grocery + interaction_parks_transit'
model_interactions = ols(formula_interactions, data=multivariate_data).fit()

# Checking for Multicollinearity using Variance Inflation Factor (VIF)
variables = model_interactions.model.exog
vif_data = pd.DataFrame()
vif_data["feature"] = model_interactions.model.exog_names
vif_data["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]

vif_data, model_interactions.summary()



# +
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Addressing Multicollinearity

# Standardizing the data for PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(multivariate_data.drop(columns=['new_deaths']))

# Applying PCA
pca = PCA()
pca_data = pca.fit_transform(scaled_data)
pca_data = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
pca_data['new_deaths'] = multivariate_data['new_deaths'].values

# Building OLS model using PCA components
formula_pca = 'new_deaths ~ ' + ' + '.join(pca_data.columns[:-1])
model_pca = ols(formula_pca, data=pca_data).fit()

# Advanced Model Diagnostics

# Residual Diagnostics
residuals = model_pca.resid

# Normality Test (Jarque-Bera)
jarque_bera_test = sm.stats.jarque_bera(residuals)

# Homoscedasticity Test (Breusch-Pagan)
bp_test = het_breuschpagan(residuals, model_pca.model.exog)

# Autocorrelation Test (Breusch-Godfrey)
bg_test = acorr_breusch_godfrey(model_pca, nlags=1)

model_pca_summary = model_pca.summary()
jarque_bera_test, bp_test, bg_test, model_pca_summary



# +
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Since ARIMA modeling is sensitive to non-stationary data, let's focus on the stationary series: COVID-19 new deaths.
# If the series is not stationary, we need to make it so (e.g., by differencing).

# Checking stationarity of COVID-19 new deaths
adf_test_deaths_stationary = adfuller(deaths_aligned)

# Proceeding with ARIMA modeling only if the series is stationary
if adf_test_deaths_stationary[1] < 0.05:
    # The series is stationary, we can proceed with ARIMA modeling
    # Setting up ARIMA model - the order (p,d,q) needs to be determined based on ACF and PACF plots or grid search
    # Here, we start with a simple model and then iterate to find the best parameters

    # Initial ARIMA Model (simple model with order (1,0,1))
    model_arima = ARIMA(deaths_aligned, order=(1,0,1))
    model_arima_fit = model_arima.fit()

    # Summary of the model
    model_arima_summary = model_arima_fit.summary()
else:
    model_arima_summary = "The series is not stationary. ARIMA modeling is not recommended without differencing or transformation."

model_arima_summary



# +
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools

# ACF and PACF plots to determine the ARIMA orders
plot_acf(deaths_aligned, lags=20)
plt.title('Autocorrelation Function')
plt.show()

plot_pacf(deaths_aligned, lags=20, method='ywm')
plt.title('Partial Autocorrelation Function')
plt.show()

# Grid Search for ARIMA orders
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))

best_aic = float("inf")
best_order = None
best_model = None

warnings.filterwarnings("ignore")  # Ignore convergence warnings

for order in pdq:
    try:
        model_temp = ARIMA(deaths_aligned, order=order)
        results_temp = model_temp.fit()
        if results_temp.aic < best_aic:
            best_aic = results_temp.aic
            best_order = order
            best_model = results_temp
    except:
        continue

warnings.resetwarnings()

best_order, best_aic, best_model.summary()



# +
# Plotting the residuals for visual inspection
plt.figure(figsize=(12, 8))

# Residuals over time for checking trends or patterns
plt.subplot(2, 1, 1)
plt.plot(residuals_arima)
plt.title('Residuals Over Time')
plt.xlabel('Date')
plt.ylabel('Residuals')

# Scatter plot of residuals against predicted values for checking homoscedasticity
plt.subplot(2, 1, 2)
predicted_values = best_model.predict()
plt.scatter(predicted_values, residuals_arima)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.tight_layout()
plt.show()


# +
# Applying Log Transformation to the Data to Achieve Normality in Residuals

# Since log transformation requires positive values, we add a constant to avoid log(0)
# Using a small constant as the data contains zeros (e.g., no deaths reported on certain days)
deaths_transformed = np.log(deaths_aligned + 1)

# Re-fitting the ARIMA(2, 0, 2) model on the transformed data
model_arima_transformed = ARIMA(deaths_transformed, order=(2,0,2))
model_arima_transformed_fit = model_arima_transformed.fit()

# Extracting the residuals from the transformed model
residuals_transformed = model_arima_transformed_fit.resid

# Residual Normality Test (Jarque-Bera) on Transformed Residuals
jarque_bera_transformed = sm.stats.jarque_bera(residuals_transformed)

# Robustness Checks and Sensitivity Analyses
# We will change one of the parameters (e.g., differencing order) and observe the impact on the model's performance
model_arima_sensitivity = ARIMA(deaths_transformed, order=(2,1,2)).fit()
residuals_sensitivity = model_arima_sensitivity.resid
jarque_bera_sensitivity = sm.stats.jarque_bera(residuals_sensitivity)

# Summary of the transformed model and sensitivity analysis results
model_arima_transformed_summary = model_arima_transformed_fit.summary()
jarque_bera_transformed, jarque_bera_sensitivity, model_arima_transformed_summary



# +
# Robustness Checks: Varying model parameters of the ARIMA model

# We will test different combinations of ARIMA parameters and observe their impact on model performance

# Varying the differencing order (d) and moving average order (q)
robustness_check_results = {}
for d in [0, 1, 2]:  # Testing differencing orders 0, 1, and 2
    for q in [1, 2, 3]:  # Testing moving average orders 1, 2, and 3
        try:
            model_temp = ARIMA(deaths_transformed, order=(2, d, q)).fit()
            aic = model_temp.aic
            ljung_box = sm.stats.acorr_ljungbox(model_temp.resid, lags=[10], return_df=True)['lb_pvalue'].iloc[0]
            robustness_check_results[(2, d, q)] = {'AIC': aic, 'Ljung-Box p-value': ljung_box}
        except:
            continue

robustness_check_results


