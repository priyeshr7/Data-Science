import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(123)

# Define the number of data points
n_samples = 1200

# Generate a skewed income distribution to be more realistic
# Using a lognormal distribution to model most people having a moderate income
income = np.random.lognormal(mean=11.5, sigma=0.6, size=n_samples)
income = np.round(income, 2)
income[income < 0] = 0

# Create the base linear relationship: as income increases, so do cars
# We'll use a simple linear model with some random noise
noise = np.random.normal(0, 0.5, size=n_samples)
number_of_cars_base = 0.00005 * income + 0.5 + noise

# Cap the number of cars to a reasonable integer value
number_of_cars = np.round(np.clip(number_of_cars_base, 0, 5)).astype(int)

# --- Add real-world scenarios (outliers) ---

# Scenario 1: Low-income individuals who own a car (e.g., inherited or older vehicles)
# Select a small number of low-income points and give them a car or two
low_income_indices = np.where(income < np.percentile(income, 20))[0]
low_income_outliers_indices = np.random.choice(low_income_indices, size=100, replace=False)
number_of_cars[low_income_outliers_indices] = np.random.randint(1, 3, size=100)

# Scenario 2: High-income individuals who own no cars (e.g., urban dwellers relying on public transport)
# Select a small number of high-income points and set their number of cars to zero
high_income_indices = np.where(income > np.percentile(income, 85))[0]
high_income_outliers_indices = np.random.choice(high_income_indices, size=80, replace=False)
number_of_cars[high_income_outliers_indices] = 0

# Create the pandas DataFrame
df = pd.DataFrame({
    'income': income,
    'number_of_cars': number_of_cars
})

# Save the DataFrame to a CSV file, including the header and without the pandas index
output_filename = 'income_vs_cars.csv'
df.to_csv(output_filename, index=False)

print(f"Dataset generated and saved successfully to '{output_filename}'.")
