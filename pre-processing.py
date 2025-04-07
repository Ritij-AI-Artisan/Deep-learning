import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create processed_data directory if not exists
os.makedirs('processed_data', exist_ok=True)

# Define file paths
files = {
    'city': 'GlobalLandTemperaturesByCity.csv',
    'country': 'GlobalLandTemperaturesByCountry.csv',
    'major_city': 'GlobalLandTemperaturesByMajorCity.csv',
    'state': 'GlobalLandTemperaturesByState.csv'
}

# Preprocessing function
def preprocess(filepath, type_name):
    print(f"Processing {type_name} data...")

    # Read the data
    df = pd.read_csv(filepath)

    # Parse date column
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')

    # Drop rows with missing AverageTemperature
    df = df.dropna(subset=['AverageTemperature'])

    # Fill missing values in other columns if needed
    df = df.fillna(method='ffill')

    # Extract Year and Month for easier analysis
    df['Year'] = df['dt'].dt.year
    df['Month'] = df['dt'].dt.month

    # Print basic info
    print(df.info())
    print(df.describe())

    # Save processed file
    output_path = os.path.join('processed_data', f'{type_name}.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned {type_name} data to {output_path}\n")

    # Quick plots
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='Year', y='AverageTemperature', data=df)
    plt.title(f"Average Temperature Over Years ({type_name.capitalize()})")
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join('processed_data', f'{type_name}_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot for {type_name} to {plot_path}\n")


# Run preprocessing for all files
for type_name, filepath in files.items():
    preprocess(filepath, type_name)

print("✅ All data processed successfully!")
