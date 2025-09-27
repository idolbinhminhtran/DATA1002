# DATA1002 - Premier League Football Analysis Project

## Project Overview
This project analyzes Premier League football data using three main data sources to explore relationships and patterns that provide insights for various stakeholders in football analytics.

## Data Sources
1. **PL_matches**: Historical match data from Premier League seasons 2015-2025
2. **ClubElo**: Club ELO ratings tracking team strength over time
3. **Understat**: Advanced football analytics data, including xG (expected goals) and other metrics

## Project Structure
```
DATA1002/
├── data/
│   ├── raw/                 # Original unprocessed data
│   ├── processed/           # Cleaned and processed data
│   └── external/            # External data references
├── notebooks/
│   ├── clean_PL.ipynb
│   ├── club_elo.ipynb
│   ├── data_analysis.ipynb
│   ├── fetch_clubelo_pl.py
│   ├── xG_premierleague_1525.ipynb
│   └── data_integration.ipynb
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
- Tutorial Session: 024
- Group Number: 01
- Group Members: Binh Minh Tran (530414672), Charlie Tran (520628478), Tuan Khai Truong (530148559)

## Submission Requirements
- Report (PDF): Final report meeting all specification requirements (Seperately on Canvas)
- Code & Data (ZIP): All notebooks, scripts, and datasets (raw + processed)
