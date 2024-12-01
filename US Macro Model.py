# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:24:29 2024

@author: user
"""
#Package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

#Variables
#CPI
#Fed Fund Rate
#Consumption
#Nasdaq

# Read excel file
file_path = r'D:\Derivatives Trading\US Macro Data.xlsx'
df = pd.read_excel(file_path)

# Convert all data to numeric first
df = df.apply(pd.to_numeric, errors='coerce')

# Create a copy of the original dataframe
df_transformed = df.copy()

# Apply log first difference to all variables
variables = ['Nasdaq', 'Consumption', 'CPI', 'Fed Fund Rate']

# First take log, then take first difference
for var in variables:
    df_transformed[var] = np.log(df[var])
    df_transformed[var] = df_transformed[var].diff()

# Select only the variables we want to use
df_transformed = df_transformed[variables]

# Remove NaN values created by differencing
df_transformed = df_transformed.dropna()

# Convert to float64 to ensure compatibility
df_transformed = df_transformed.astype('float64')

# Create and fit the VAR model
model = VAR(df_transformed)
var_model = model.fit(maxlags=4)

# Get roots and calculate number outside unit circle
roots = var_model.roots
abs_roots = np.abs(roots)
n_outside = sum(abs_roots >= 1)
is_stable = n_outside == 0

# Print detailed stability analysis
print("\nStability Analysis Results:")
print("==========================")
print(f"Total number of roots: {len(roots)}")
print(f"Number of roots outside unit circle: {n_outside}")
print(f"VAR satisfies stability condition: {is_stable}")
print("\nModulus of each root:")
for i, modulus in enumerate(abs_roots, 1):
    print(f"Root {i}: {modulus:.6f}")

# Create the plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

# Draw unit circle
circle = plt.Circle((0,0), 1, fill=False, color='black')
ax.add_artist(circle)

# Plot roots
ax.scatter(roots.real, roots.imag, color='blue', marker='o')

# Add reference lines
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Set plot properties with more detailed title
stability_text = "STABLE" if is_stable else "NOT STABLE"
title = f'Inverse Roots of AR Characteristic Polynomial\n'
title += f'Number of roots outside unit circle: {n_outside}\n'
title += f'VAR Model is {stability_text}'

ax.grid(True)
ax.set_xlabel('Real part')
ax.set_ylabel('Imaginary part')
ax.set_title(title)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')

plt.show()

########################################################VAR#########################################
# Generate impulse response functions
irf = var_model.irf(periods=24)  # 24 periods ahead

# Plot impulse responses to a shock in Fed Fund Rate
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Impulse Responses to a Shock in Fed Fund Rate')

# Get variable positions
ffr_pos = variables.index('Fed Fund Rate')

for i, var in enumerate(variables):
    row = i // 2
    col = i % 2
    
    # Get impulse response data
    response_data = irf.irfs[:, i, ffr_pos]
    periods = range(len(response_data))
    
    # Plot impulse response
    axes[row, col].plot(periods, response_data)
    axes[row, col].set_title(f'Response of {var} to Fed Fund Rate Shock')
    axes[row, col].grid(True)
    axes[row, col].axhline(y=0, color='r', linestyle='--')
    axes[row, col].set_xlabel('Periods')
    axes[row, col].set_ylabel('Response')

plt.tight_layout()
plt.show()

# Print numerical values of impulse responses
print("\nNumerical Values of Impulse Responses to Fed Fund Rate Shock:")
print("========================================================")
for i, var in enumerate(variables):
    print(f"\nResponse of {var}:")
    print("Period    Response")
    print("-------------------")
    response_data = irf.irfs[:, i, ffr_pos]
    for period in range(10):  # Print first 10 periods
        print(f"{period:<8} {response_data[period]:>.6f}")

# Calculate and plot cumulative responses
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Cumulative Impulse Responses to a Shock in Fed Fund Rate')

for i, var in enumerate(variables):
    row = i // 2
    col = i % 2
    
    # Calculate cumulative response
    response_data = irf.irfs[:, i, ffr_pos]
    cumulative_response = np.cumsum(response_data)
    periods = range(len(cumulative_response))
    
    # Plot cumulative impulse response
    axes[row, col].plot(periods, cumulative_response)
    axes[row, col].set_title(f'Cumulative Response of {var} to Fed Fund Rate Shock')
    axes[row, col].grid(True)
    axes[row, col].axhline(y=0, color='r', linestyle='--')
    axes[row, col].set_xlabel('Periods')
    axes[row, col].set_ylabel('Cumulative Response')

plt.tight_layout()
plt.show()

# Print forecast error variance decomposition
print("\nForecast Error Variance Decomposition:")
print("=====================================")
fevd = var_model.fevd(periods=24)
print(fevd.summary())