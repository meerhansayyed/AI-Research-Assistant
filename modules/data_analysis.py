import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_csv(file_path):
    """
    Reads a CSV file and returns basic statistics and column information.
    Creates histograms for numerical columns and saves analysis results to CSV files.
    """
    df = pd.read_csv(file_path)
    
    # Basic dataset info
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_stats": df.describe().to_dict(),
        "summary_csv": "assets/analysis/summary_statistics.csv",
        "missing_csv": "assets/analysis/missing_values.csv"
    }
    
    # Create folders if not exist
    os.makedirs("assets/charts", exist_ok=True)
    os.makedirs("assets/analysis", exist_ok=True)
    
    charts = []
    
    # Plot histogram for numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=20)
        plt.title(f"Distribution of {col}")
        chart_path = f"assets/charts/{col}_hist.png"
        plt.savefig(chart_path)
        plt.close()
        charts.append(chart_path)
    
    # Save summary stats to CSV
    summary_df = df.describe().transpose()
    summary_df.to_csv(info["summary_csv"])
    
    # Save missing values info to CSV
    missing_df = pd.DataFrame(list(info["missing_values"].items()), columns=["Column", "MissingValues"])
    missing_df.to_csv(info["missing_csv"], index=False)
    
    # ✅ Now returns two values as expected by app.py
    return info, charts
