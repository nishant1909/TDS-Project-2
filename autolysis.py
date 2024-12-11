import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI

# Initialize OpenAI
openai_api_key = os.getenv("AIPROXY_TOKEN")
if not openai_api_key:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

def load_data(file_path):
    """Load the dataset from a CSV file."""
    encodings = ["utf-8", "ISO-8859-1", "Windows-1252"]
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns using {encoding} encoding.")
            return data
        except UnicodeDecodeError:
            print(f"Encoding {encoding} failed. Trying next...")
        except Exception as e:
            print(f"Error loading data with {encoding} encoding: {e}")
    print("Error: Unable to load the dataset with tried encodings.")
    sys.exit(1)


def basic_analysis(data):
    """Perform basic statistical analysis."""
    analysis = {
        "Summary Statistics": data.describe(include='all').to_dict(),
        "Missing Values": data.isnull().sum().to_dict(),
        "Data Types": data.dtypes.apply(str).to_dict(),
    }
    return analysis

def visualize_correlation(data, output_dir, dataset_name):
    """Generate a correlation heatmap."""
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if numeric_data.empty:
        print("No numeric data available for correlation analysis.")
        return
    corr = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    output_file = os.path.join(output_dir, f"{dataset_name}_heatmap.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Correlation heatmap saved to {output_file}.")

def query_llm(prompt):
    """Query the LLM for insights or code."""
    try:
        response = OpenAI("gpt-4o-mini", api_key=openai_api_key).call(prompt)
        return response
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return ""

def narrate_story(data_info, analysis, insights):
    """Generate a narrative from the analysis."""
    prompt = f"""
You are a data analyst assistant. Write a Markdown story for the following data analysis:

Dataset Information:
{data_info}

Analysis Summary:
{analysis}

Key Insights:
{insights}

Please include headers and structure it well for a README.md file.
"""
    return query_llm(prompt)

def save_story(story, output_dir):
    """Save the narrative to a README.md file."""
    output_file = os.path.join(output_dir, "README.md")
    with open(output_file, "w") as f:
        f.write(story)
    print(f"Narrative saved to {output_file}.")

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.getcwd(), dataset_name)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    data = load_data(file_path)

    # Basic analysis
    analysis = basic_analysis(data)

    # Visualization
    visualize_correlation(data, output_dir, dataset_name)

    # Generate story
    data_info = {
        "Columns": data.columns.tolist(),
        "Shape": data.shape,
    }
    insights = "The analysis provided insights into data patterns and correlations."
    story = narrate_story(data_info, analysis, insights)

    # Save results
    save_story(story, output_dir)

if __name__ == "__main__":
    main()
