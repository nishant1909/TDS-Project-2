
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
import requests
import json

# Function to analyze the data (summary stats, missing values, correlation matrix)
def analyze_data(df):
    """Analyze the dataset to extract summary statistics, missing values, and correlation matrix."""
    summary_stats = df.describe()
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
    """Generate and save visualizations: correlation heatmap, outliers, and distribution plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Correlation heatmap
    if not corr_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()

    # Outliers visualization
    if not outliers.empty and outliers.sum() > 0:
        outliers.plot(kind='bar', color='red', figsize=(10, 6))
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        plt.savefig(os.path.join(output_dir, 'outliers.png'))
        plt.close()

    # Distribution plot for the first numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if numeric_columns.any():
        plt.figure(figsize=(10, 6))
        sns.histplot(df[numeric_columns[0]], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {numeric_columns[0]}')
        plt.savefig(os.path.join(output_dir, 'distribution.png'))
        plt.close()

# Function to create a README.md file
def create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir):
    """Generate a markdown report summarizing analysis and insights."""
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Summary Statistics\n")
        f.write(summary_stats.to_markdown())
        f.write("\n\n## Missing Values\n")
        f.write(missing_values.to_markdown())
        f.write("\n\n## Correlation Matrix\n")
        if not corr_matrix.empty:
            f.write("![Correlation Matrix](correlation_matrix.png)\n\n")
        f.write("## Outliers\n")
        f.write(outliers.to_markdown())
        f.write("\n\n## Visualizations\n")
        f.write("![Outliers](outliers.png)\n\n![Distribution](distribution.png)\n")
    return readme_path

# Function to interact with OpenAI API for generating narratives
def generate_story(context, prompt):
    """Generate a story using OpenAI API."""
    try:
        token = os.getenv("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\nPrompt: {prompt}"}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        return response.json()['choices'][0]['message']['content'] if response.status_code == 200 else "Error generating story."
    except Exception as e:
        return f"Error: {e}"

# Main function
def main(csv_path, output_dir):
    """Main function to orchestrate data analysis, visualization, and reporting."""
    df = pd.read_csv(csv_path)
    summary_stats, missing_values, corr_matrix = analyze_data(df)
    outliers = detect_outliers(df)
    visualize_data(corr_matrix, outliers, df, output_dir)

    context = f"Summary Statistics: {summary_stats}\nMissing Values: {missing_values}\nOutliers: {outliers}"  # Concise prompt
    story = generate_story(context, "Generate a detailed analysis narrative.")

    readme_path = create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir)
    with open(readme_path, 'a') as f:
        f.write("\n\n## Generated Story\n")
        f.write(story)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated Data Analysis Script")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("output_dir", nargs='?', default="results", help="Directory to save outputs (default: 'results')")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    main(args.csv_path, args.output_dir)
