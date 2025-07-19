## Tourism and Economic Growth — Insights from the TEI Dataset

### Table of Contents
- Introduction
- Data Wrangling
- Exploratory Data Analysis (EDA)
- Result Interpretation
- Conclusion
- Limitation

### Introduction

Tourism has remained major contributor to a country’s Gross Domestic Product (GDP) over time. In 2024, tourism was estimated to contribute over 10% to global GDP through the creation of employment opportunities.

The tourism industry has consistently grown over the years. In 2024 alone, it provided jobs for over 348 million people worldwide, accounting for 1 in every 10 jobs globally.

The dataset we intend to analyze contains information on key tourism activities and economic indicators for over 200 countries, covering the period from 1999 to 2023. The dataset includes the following variables:

Tourism Receipts: Total income a country generates from international tourism.

Tourism Arrivals: Total number of international tourists entering a country.

Tourism Expenditures: Amount of money spent by international tourists within a country.

Tourism Exports: The percentage of a country’s total exports derived from international tourism receipts.

Tourism Departures: Number of citizens or residents traveling abroad for tourism.

GDP: Total value of all goods and services produced in a country.

Inflation: Annual percentage change in the average price of goods and services in a country.

Unemployment: Percentage of people within the labor force who are unemployed but actively seeking work.

All currency values are expressed in current US dollars.

Understanding the various factors influencing economic growth — and exploring the relationship between tourism and economic performance — is crucial in today’s global economy. This knowledge can help individuals, businesses, and governments make informed decisions. The tourism sector generates significant revenue and supports many industries, including retail, hospitality, financial services, and especially the transportation sector.

In this exploratory data analysis (EDA), we will address several key questions using the Tourism and Economic Impact (TEI) Dataset, sourced from the World Bank (via Kaggle). These include:

1. What is the relationship between tourism receipts and economic indicators such as inflation, unemployment, tourism expenditures, tourism arrivals, etc.?

2. What is the overall impact of tourism on economic growth?



```
##Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

### Data Wrangling
# Here, the data is loaded for preprocessing and cleaning
data = pd.read_csv(r'C:\Users\HP\Desktop\End-to-End-Project\world_tourism_economy_data.csv')
data.head()


