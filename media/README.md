# Project: Data Analysis of Media Dataset

## Overview

This project performs a data analysis of a media dataset and provides an understanding of the relationships between different variables such as **overall**, **quality**, and **repeatability**. Using statistical analysis, a correlation matrix is visualized to identify patterns between the variables.

The analysis includes:

- Loading the dataset
- Calculating the correlation matrix
- Visualizing the matrix using a heatmap for better understanding of the relationships

## How Data Was Scraped and Processed

The data for this analysis is stored in a CSV file, **media.csv**, which contains multiple attributes relevant to the analysis. The dataset is processed by first reading it into a DataFrame and then calculating the correlation between different columns of interest (i.e., **overall**, **quality**, and **repeatability**).

The correlation matrix is then plotted using a heatmap to visually interpret the relationships between these variables.

### Steps:

1. Load the dataset from a CSV file.
2. Calculate the correlation between the specified columns.
3. Create a heatmap to visually represent the correlation matrix.

## Interesting Findings

- The variables **overall** and **quality** have a high correlation of **0.83**, indicating a strong relationship between these two factors.
- **Overall** and **repeatability** show a moderate correlation of **0.51**, suggesting some relationship, though weaker.
- The correlation between **quality** and **repeatability** is **0.31**, implying a weaker relationship.

## Actionable Recommendation for Developers

- Developers working with media data can use this correlation analysis to prioritize **overall** and **quality** in their models, as they have the strongest relationship.
- **Repeatability** can still be a valuable factor, but it may not need to be as heavily weighted when compared to **overall** and **quality** based on this analysis.

## Heatmap

The following heatmap visualizes the correlation matrix between **overall**, **quality**, and **repeatability**:

![Correlation Heatmap](media_heatmap.png)

## Files

- **media.csv**: The dataset used for analysis.
- **media_heatmap.png**: The heatmap generated from the correlation matrix.
