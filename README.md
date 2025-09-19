# DATA1002 - Premier League Football Analysis Project

## Project Overview
This project analyzes Premier League football data using three main data sources to explore relationships and patterns that provide insights for various stakeholders in football analytics.

## Data Sources
1. **PL_matches**: Historical match data from Premier League seasons 2015-2025
2. **ClubElo**: Club ELO ratings tracking team strength over time
3. **Understat**: Advanced football analytics data including xG (expected goals) and other metrics

## Project Structure
```
DATA1002/
├── data/
│   ├── raw/                 # Original unprocessed data
│   ├── processed/           # Cleaned and processed data
│   └── external/            # External data references
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_analysis.ipynb
│   └── 04_main_report.ipynb
├── src/                     # Python modules and helper functions
├── reports/                 # Generated reports and submissions
├── figures/                 # Saved plots and visualizations
├── tests/                   # Unit tests for data validation
├── docs/                    # Documentation and data dictionaries
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. Start with `notebooks/01_data_exploration.ipynb` to understand the datasets
2. Run `notebooks/02_data_cleaning.ipynb` for data preparation
3. Perform analysis in `notebooks/03_analysis.ipynb`
4. View the complete report in `notebooks/04_main_report.ipynb`

## Group Information
- Course: DATA1002 - Informatics: Data and Computation
- Tutorial Section: [TO BE FILLED]
- Group Number: [TO BE FILLED]
- Group Members: [TO BE FILLED]

## Submission Requirements
- Report (PDF): Final report meeting all specification requirements
- Code & Data (ZIP): All notebooks, scripts, and datasets (raw + processed)
