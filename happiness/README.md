# Project: Global Happiness Analysis

## Overview

This project analyzes a dataset related to global happiness and explores relationships between various factors, such as **Life Ladder**, **GDP per capita**, **Social support**, **Life expectancy at birth**, **Freedom to make life choices**, **Perceptions of corruption**, **Generosity**, **Positive affect**, and **Negative affect**.

Using statistical analysis, the correlation matrix is computed and visualized to understand the relationships between these variables.

The analysis includes:

- Loading the dataset
- Calculating the correlation matrix for key features
- Visualizing the correlation matrix using a heatmap to identify patterns

## How Data Was Scraped and Processed

The dataset, **happiness.csv**, contains various attributes representing economic, social, and psychological aspects of happiness worldwide. The data was loaded into a DataFrame for statistical analysis, focusing on the correlations between the selected features.

### Steps:

1. Load the dataset from the CSV file.
2. Calculate the correlation between the features of interest.
3. Create a heatmap to visually represent the correlations and identify meaningful patterns.

## Interesting Findings

- **Life Ladder** has a strong positive correlation with **GDP per capita** (0.78) and **Social support** (0.72), indicating these factors significantly influence overall happiness.
- **GDP per capita** is highly correlated with **Life expectancy at birth** (0.82), suggesting economic prosperity aligns with better health outcomes.
- **Freedom to make life choices** is positively correlated with **Positive affect** (0.58), emphasizing the psychological impact of autonomy.
- **Perceptions of corruption** has a negative correlation with **Life Ladder** (-0.43), suggesting that higher corruption perceptions are linked to lower happiness levels.

## Actionable Recommendations

- Policymakers should prioritize **GDP growth**, **social support systems**, and **life expectancy improvements** to boost happiness in their regions.
- Efforts to increase **freedom to make life choices** can enhance emotional well-being and overall happiness.
- Addressing corruption and improving perceptions of governance may lead to significant improvements in happiness levels.

## Heatmap

The following heatmap visualizes the correlation matrix of the key features:

![Happiness Correlation Heatmap](media_heatmap.png)

## Files

- **happiness.csv**: The dataset used for the analysis.
- **media_heatmap.png**: The heatmap generated from the correlation matrix.

## Tools and Libraries Used

- **Python**: For data loading and processing
- **Pandas**: To handle the CSV dataset
- **Seaborn**: For generating the correlation heatmap
- **Matplotlib**: For visualizations
