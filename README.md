# Premier League Match Outcome Prediction Analysis (2015-2025)

## Research Question
**How do key pre-match factors influence Premier League match outcomes across the 2015–2025 seasons?**

## Project Overview
This project analyzes 10 years of Premier League data (2015-2025) to understand how pre-match factors such as ELO ratings, expected goals (xG), and betting odds influence match outcomes. Through comprehensive data analysis and visualization, we evaluate the predictive power of these factors and provide insights into match outcome patterns.

## Key Findings
- **Expected Goals (xG)** demonstrates the highest prediction accuracy (61.2%) among all methods
- **ELO ratings** show consistent performance (53.2% accuracy) with strong correlation to match outcomes
- **Home advantage** remains significant, with home teams winning 44.4% of all matches
- All prediction methods significantly outperform random chance (33.3% baseline)

## Project Structure
```
DATA1002/
├── config/                     # Configuration files
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── PL_matches/         # Premier League match results (2015-2025)
│   │   │   ├── 2015-2016.csv
│   │   │   ├── 2016-2017.csv
│   │   │   ├── 2017-2018.csv
│   │   │   ├── 2018-2019.csv
│   │   │   ├── 2019-2020.csv
│   │   │   ├── 2020-2021.csv
│   │   │   ├── 2021-2022.csv
│   │   │   ├── 2022-2023.csv
│   │   │   ├── 2023-2024.csv
│   │   │   └── 2024-2025.csv
│   │   ├── xG/                 # Expected goals data (2015-2025)
|   |   |   └── xG_premierleague.csv 
│   │   └── Club Elo/           # ELO ratings history
│   │       └── clubelo_premierleague_history.csv
│   └── processed/              # Cleaned and integrated datasets
│       ├── PL_integrated_dataset_10years.csv     # Main integrated dataset
│       ├── PL_matches_10years_cleaned.csv        # Cleaned match data*
│       ├── PL_xG_10years_understat.csv           # Cleaned xG data from Understat*
│       ├── PL_xG_trends_by_season_2015_2025.csv  # xG trends analysis
│       ├── PL_outcomes_by_season_2015_2025.csv   # Season-wise outcomes
│       └── clubelo_premierleague_history.csv     # Cleaned ELO ratings data*
├── notebooks/
│   ├── clean_PL.ipynb          # PL match results cleaning and preprocessing*
│   ├── data_integration.ipynb  # Dataset integration and feature engineering
│   ├── data_analysis.ipynb     # PL match results analysis and visualizations*
│   ├── club_elo.ipynb          # Club ELO ratings analysis and visualizations*
│   ├── xG_premierleague_1525.ipynb  # xG data cleaning and processing*
│   ├── xG_premierleague_1525_analysis.ipynb  # xG data analysis and visualizations*
│   └── fetch_clubelo_pl.py     # Club ELO ratings cleaning and preprocessing*
├── figures/                    # Generated visualizations
│   ├── summary.png             # Summary dashboard
│   ├── output.png              # Main results visualization
│   ├── xG_vs_output.png        # xG analysis results
│   ├── elo_distribution.png    # ELO ratings distribution
│   ├── elo_bin_dif.png         # ELO difference bins analysis
│   ├── mean_prediction_success_bar_chart.png    # Prediction success rates
│   └── prediction_success_home_vs_away.png # Home/Away predictions
├── docs/                       # Documentation directory
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

## Methodology

### Data Sources
1. **Premier League Match Data**: Historical match results from football-data.co.uk
2. **ELO Ratings**: Team strength ratings from clubelo.com
3. **Expected Goals (xG)**: Match-level xG statistics from Understat

### Pre-Match Factors Analyzed
1. **ELO Ratings**: Team strength indicators based on historical performance
2. **Expected Goals (xG)**: Statistical measure of scoring opportunities
3. **Betting Odds**: Market predictions from major bookmakers (Bet365)
4. **Home Advantage**: Impact of playing at home stadium

### Analysis Approach
1. **Data Integration**: Combined multiple datasets using match date and team names
2. **Feature Engineering**: Created derived features (ELO difference, xG difference)
3. **Binary Classification**: Analyzed win/lose predictions excluding draws
4. **Statistical Evaluation**: Calculated accuracy, precision, recall, and confidence intervals
5. **Visualization**: Created comprehensive dashboards for insights

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Git

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/idolbinhminhtran/DATA1002.git
cd DATA1002

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries
```python
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scipy==1.10.1
scikit-learn==1.2.2
jupyter==1.0.0
```

## Usage

### 1. Data Preparation
```bash
# Run data cleaning notebook
jupyter notebook notebooks/clean_PL.ipynb

# Run data integration notebook  
jupyter notebook notebooks/data_integration.ipynb
```

### 2. Run Analysis
```bash
# Open main analysis notebook
jupyter notebook notebooks/data_analysis.ipynb
```

### 3. Key Notebooks

#### `clean_PL.ipynb`
- Cleans raw Premier League match data
- Handles missing values and inconsistencies
- Standardizes team names and date formats
- Outputs: `PL_matches_10years_cleaned.csv`

#### `data_integration.ipynb`
- Merges match results with ELO ratings and xG data
- Creates derived features for analysis
- Integrates multiple data sources into unified dataset
- Outputs: `PL_integrated_dataset_10years.csv`

#### `data_analysis.ipynb`
- Main analysis notebook with all visualizations
- Binary win/lose prediction analysis
- Correlation analysis between pre-match factors and outcomes
- Professional visualization dashboard
- Generates all figures in `figures/` directory

#### `club_elo.ipynb`
- Analyzes ELO ratings distribution and trends
- Visualizes team strength over time
- Creates ELO-based predictions

#### `xG_premierleague_1525.ipynb`
- Collects and processes xG data from Understat
- Analyzes xG trends across seasons
- Creates xG-based match predictions
- Outputs: `PL_xG_10years_understat.csv`, `PL_xG_trends_by_season_2015_2025.csv`

#### `fetch_clubelo_pl.py`
- Python script to fetch latest Club ELO ratings
- Automates data collection from clubelo.com
- Can be run independently for data updates

## Results & Visualizations

### Key Visualizations Generated
1. **Prediction Accuracy Comparison**: Compares ELO, xG, and betting odds accuracy
2. **Correlation Heatmaps**: Shows relationships between predictions and actual outcomes
3. **Confidence Interval Plots**: Statistical significance of prediction methods
4. **Bias Analysis**: Reveals systematic biases in each prediction method

### Main Insights
1. **xG Performance**: Most accurate predictor, especially for high-confidence predictions
2. **ELO Stability**: Consistent performance across different match contexts
3. **Home Advantage**: Persistent factor with ~12% advantage for home teams
4. **Prediction Biases**: All methods underestimate draw occurrences

## Reproducibility

### Data Availability
All processed datasets are included in `data/processed/`. Raw data sources:
- Premier League matches: [football-data.co.uk](https://www.football-data.co.uk/)
- ELO ratings: [clubelo.com](http://clubelo.com/)
- xG data: [Understat](https://understat.com/)

### Running the Analysis
1. Ensure all dependencies are installed
2. Run notebooks in order: cleaning → integration → analysis
3. All outputs will be saved to respective directories

## Future Work
- Incorporate additional features (weather, injuries, team form)
- Develop ensemble prediction models
- Extend analysis to other leagues
- Real-time prediction system implementation


## License
This project is for educational purposes as part of the DATA1002 course at the University of Sydney.

## Acknowledgments
- Football-data.co.uk for match data
- Clubelo.com for ELO ratings
- Understat for xG statistics
- DATA1002 course instructors for guidance
