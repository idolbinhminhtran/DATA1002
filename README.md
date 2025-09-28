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
├── README.md
├── .gitattributes
├── .gitignore
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── PL_matches/         # Premier League match results (2015-2025)
│   │   ├── xG/                 # Expected goals data
│   │   └── Club Elo/           # ELO ratings history
│   └── processed/              # Cleaned and integrated datasets
│       ├── PL_integrated_dataset_10years.csv
│       ├── PL_matches_10years_cleaned.csv
│       └── PL_xG_10years_understat.csv
├── notebooks/
│   ├── clean_PL.ipynb          # Data cleaning and preprocessing
│   ├── data_integration.ipynb  # Dataset integration and feature engineering
│   ├── data_analysis.ipynb     # Main analysis and visualizations
│   └── xG_premierleague_1525.ipynb  # xG data collection
├── figures/                    # Generated visualizations
├── report/                    # Analysis reports
|   ├── DATA1002_Assignment_Stage1.pdf
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

#### `data_integration.ipynb`
- Merges match results with ELO ratings and xG data
- Creates derived features for analysis
- Exports integrated dataset

#### `data_analysis.ipynb`
- Main analysis notebook with all visualizations
- Binary win/lose prediction analysis
- Correlation analysis between pre-match factors and outcomes
- Professional visualization dashboard

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
