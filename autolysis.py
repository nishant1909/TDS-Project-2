
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "ipykernel",  # Added ipykernel
# ]
# ///


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai

# Function to analyze the data (basic summary stats, missing values, correlation matrix)
def analyze_data(df):
    """Analyze dataset and return summary statistics, missing values, and correlation matrix."""
    summary_stats = df.describe(include='all').transpose()
    missing_values = df.isnull().sum()
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
    return summary_stats, missing_values, corr_matrix

# Function to detect outliers using the IQR method
def detect_outliers(df):
    """Detect outliers in numeric columns using the IQR method."""
    df_numeric = df.select_dtypes(include=[np.number])
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
    return outliers

# Function to generate visualizations
def visualize_data(corr_matrix, outliers, df, output_dir):
    """Generate visualizations for correlation matrix, outliers, and distributions."""
    output_files = {}

    # Correlation heatmap
    if not corr_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        heatmap_file = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(heatmap_file)
        plt.close()
        output_files['correlation_matrix'] = heatmap_file

    # Outliers bar plot
    if not outliers.empty and outliers.sum() > 0:
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        outliers_file = os.path.join(output_dir, 'outliers.png')
        plt.savefig(outliers_file)
        plt.close()
        output_files['outliers'] = outliers_file

    # Distribution plot for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {column}')
        dist_file = os.path.join(output_dir, f'distribution_{column}.png')
        plt.savefig(dist_file)
        plt.close()
        output_files[f'distribution_{column}'] = dist_file

    return output_files

# Function to create README.md with analysis and narrative
def create_readme(summary_stats, missing_values, corr_matrix, outliers, visualizations, output_dir):
    """Create a README file summarizing the analysis and including visualizations."""
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Automated Data Analysis Report\n\n")

        # Summary statistics
        f.write("## Summary Statistics\n")
        f.write(summary_stats.to_markdown() + "\n\n")

        # Missing values
        f.write("## Missing Values\n")
        f.write(missing_values.to_markdown() + "\n\n")

        # Correlation matrix
        if 'correlation_matrix' in visualizations:
            f.write("## Correlation Matrix\n")
            f.write(f"![Correlation Matrix]({visualizations['correlation_matrix']})\n\n")

        # Outliers
        if 'outliers' in visualizations:
            f.write("## Outliers\n")
            f.write(f"![Outliers]({visualizations['outliers']})\n\n")

        # Distributions
        for key, path in visualizations.items():
            if key.startswith('distribution_'):
                f.write(f"## {key.capitalize()}\n")
                f.write(f"![{key.capitalize()}]({path})\n\n")

    return readme_path

# Function to generate a narrative using OpenAI
def generate_narrative(prompt):
    """Generate a narrative using OpenAI's API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating narrative: {e}"

# Main function
def main(csv_file, output_dir):
    df = pd.read_csv(csv_file)
    summary_stats, missing_values, corr_matrix = analyze_data(df)
    outliers = detect_outliers(df)
    visualizations = visualize_data(corr_matrix, outliers, df, output_dir)
    readme_path = create_readme(summary_stats, missing_values, corr_matrix, outliers, visualizations, output_dir)

    narrative_prompt = f"""
    Based on the following data:
    - Summary Statistics: {summary_stats.to_dict()}
    - Missing Values: {missing_values.to_dict()}
    - Correlation Matrix: {corr_matrix.to_dict()}
    - Outliers: {outliers.to_dict()}

    Generate a detailed narrative including insights and implications.
    """
    narrative = generate_narrative(narrative_prompt)
    with open(os.path.join(output_dir, 'Narrative.md'), 'w') as f:
        f.write(narrative)

    print(f"README and narrative generated in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Data Analysis Script")
    parser.add_argument('csv_file', help="Path to the CSV file")
    parser.add_argument('--output_dir', default='output', help="Directory to save outputs")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args.csv_file, args.output_dir)
