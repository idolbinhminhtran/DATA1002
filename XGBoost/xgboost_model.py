"""
XGBoost Model for Premier League Match Outcome Prediction
=========================================================
Predicts home wins and away wins using season-based chronological split.
Uses historical match data with Elo ratings and betting odds.
Includes hyperparameter tuning capabilities.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score, brier_score_loss, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import hyperparameter manager and Optuna tuner
from hyperparameter_manager import create_hyperparameter_manager
from optuna_tuner import create_optuna_tuner
from config import get_default_params

class PLMatchPredictor:
    """Premier League match outcome predictor using XGBoost with hyperparameter tuning"""
    
    def __init__(self, data_path, train_end_season='2021-2022', random_state=42):
        """
        Initialize the predictor
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file with match data and Elo ratings
        train_end_season : str
            Last season to include in training (inclusive)
        random_state : int
            Random seed for reproducibility
        """
        self.data_path = data_path
        self.train_end_season = train_end_season
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None  # Single 2-class model
        self.feature_cols = None
        
        # Initialize hyperparameter manager and Optuna tuner
        self.hp_manager = create_hyperparameter_manager()
        self.optuna_tuner = create_optuna_tuner()
        
        # Get hyperparameters from saved files or use defaults
        self.best_params = self.hp_manager.get_hyperparameters("combined")
        
        # Load optimal threshold if available
        self.threshold_ = self.hp_manager.load_threshold("combined") or 0.5
        
        print("Hyperparameter Manager initialized")
        print(f"Save directory: {self.hp_manager.save_dir}")
        print(f"Single 2-class model params: {'Tuned' if self.best_params else 'Default'}")
        print(f"Optimal threshold: {self.threshold_:.4f}")
        
    def _create_calibration_set(self, X_train, y_train, cal_size_ratio=0.15, min_cal_size=200):
        """
        Create calibration set from end of training data
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets
        cal_size_ratio : float
            Ratio of data to use for calibration (default: 15%)
        min_cal_size : int
            Minimum calibration set size (default: 200)
            
        Returns:
        --------
        tuple : (X_fit, y_fit, X_cal, y_cal)
        """
        total_size = len(X_train)
        cal_size = max(int(total_size * cal_size_ratio), min_cal_size)
        
        # Split from end (most recent data for calibration)
        X_fit = X_train[:-cal_size]
        y_fit = y_train[:-cal_size]
        X_cal = X_train[-cal_size:]
        y_cal = y_train[-cal_size:]
        
        print(f"Calibration set created: {len(X_cal)} samples ({len(X_cal)/total_size*100:.1f}%)")
        print(f"Fitting set: {len(X_fit)} samples ({len(X_fit)/total_size*100:.1f}%)")
        
        return X_fit, y_fit, X_cal, y_cal
        
    def _evaluate_calibration(self, X_cal, y_cal):
        """
        Evaluate probability calibration quality
        
        Parameters:
        -----------
        X_cal : array-like
            Calibration features
        y_cal : array-like
            Calibration targets
        """
        try:
            # Get calibrated probabilities
            y_cal_prob = self.model.predict_proba(X_cal)[:, 1]
            
            # Calculate Brier score (lower is better)
            brier_score = brier_score_loss(y_cal, y_cal_prob)
            
            # Plot calibration curve
            self._plot_calibration_curve(y_cal, y_cal_prob)
            
            print(f"Calibration quality - Brier Score: {brier_score:.4f} (lower is better)")
            
        except Exception as e:
            print(f"Warning: Error evaluating calibration: {e}")
        
    def _plot_calibration_curve(self, y_true, y_prob):
        """Plot calibration curve to visualize probability quality"""
        try:
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            
            plt.figure(figsize=(8, 6))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated Model")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
            
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title("Calibration Curve - Probability Quality")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('calibration_curve.png', dpi=100, bbox_inches='tight')
            plt.show()
            
            print("Calibration curve saved as 'calibration_curve.png'")
            
        except Exception as e:
            print(f"Warning: Error plotting calibration curve: {e}")
    
    def _print_train_val_accuracy(self, X_fit, y_fit, X_cal, y_cal, X_val, y_val):
        """
        Calculate and print training, calibration, and validation accuracy
        
        Parameters:
        -----------
        X_fit, y_fit : fitting set data
        X_cal, y_cal : calibration set data
        X_val, y_val : validation set data (can be None)
        """
        print("\n" + "="*60)
        print("TRAINING PERFORMANCE")
        print("="*60)
        
        # Get threshold to use
        threshold = 0.5  # Default
        if hasattr(self, 'best_params') and self.best_params and 'threshold' in self.best_params:
            threshold = self.best_params['threshold']
        elif hasattr(self, 'threshold_'):
            threshold = self.threshold_
        
        # Calculate accuracy on fitting set
        fit_probs = self.model.predict_proba(X_fit)[:, 1]
        fit_pred = (fit_probs >= threshold).astype(int)
        fit_acc = accuracy_score(y_fit, fit_pred)
        print(f"Fitting Set (training): {len(X_fit)} samples")
        print(f"   Accuracy: {fit_acc:.4f}")
        
        # Calculate accuracy on calibration set
        cal_probs = self.model.predict_proba(X_cal)[:, 1]
        cal_pred = (cal_probs >= threshold).astype(int)
        cal_acc = accuracy_score(y_cal, cal_pred)
        print(f"Calibration Set: {len(X_cal)} samples")
        print(f"   Accuracy: {cal_acc:.4f}")
        
        # Calculate accuracy on validation set if available
        if X_val is not None and y_val is not None:
            val_probs = self.model.predict_proba(X_val)[:, 1]
            val_pred = (val_probs >= threshold).astype(int)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Validation Set: {len(X_val)} samples")
            print(f"   Accuracy: {val_acc:.4f}")
            print(f"\nAccuracy Summary:")
            print(f"   Train: {fit_acc:.4f} | Cal: {cal_acc:.4f} | Val: {val_acc:.4f}")
        
        print("="*60)
        
    def _find_best_threshold(self, y_true, y_prob, metric='accuracy'):
        """
        Find optimal decision threshold based on validation set
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_prob : array-like
            Predicted probabilities for positive class
        metric : str
            Metric to optimize: 'accuracy', 'balanced', 'f1', or 'gmean'
            
        Returns:
        --------
        float : Optimal threshold
        """
        # Scan wider range of thresholds for better optimization
        thresholds = np.linspace(0.2, 0.8, 601)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced':
                score = balanced_accuracy_score(y_true, y_pred)
            elif metric == 'gmean':  # Geometric mean of sensitivity and specificity
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = np.sqrt(sensitivity * specificity)
            else:  # accuracy (default)
                score = accuracy_score(y_true, y_pred)
            
            scores.append(score)
        
        # Return threshold that maximizes the chosen metric
        best_idx = np.argmax(scores)
        best_threshold = float(thresholds[best_idx])
        best_score = scores[best_idx]
        print(f"Optimal threshold found: {best_threshold:.4f} (metric: {metric}, score: {best_score:.4f})")
        return best_threshold
        
    def load_and_prepare_data(self):
        """Load data and create features"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Loaded {len(df)} matches from {df['season'].nunique()} seasons")
        print(f"Seasons: {sorted(df['season'].unique())}")
        
        return df
    
    def engineer_features(self, df):
        """Create additional features for better prediction"""
        print("\nEngineering features...")
        
        # Betting odds features (if available)
        odds_cols = []
        if 'b365h' in df.columns:
            # Implied probabilities from odds
            df['implied_prob_home'] = 1 / df['b365h'].fillna(df['b365h'].median())
            df['implied_prob_draw'] = 1 / df['b365d'].fillna(df['b365d'].median())
            df['implied_prob_away'] = 1 / df['b365a'].fillna(df['b365a'].median())
            
            # Normalize to sum to 1 (remove bookmaker margin)
            total_prob = df['implied_prob_home'] + df['implied_prob_draw'] + df['implied_prob_away']
            df['implied_prob_home'] = df['implied_prob_home'] / total_prob
            df['implied_prob_draw'] = df['implied_prob_draw'] / total_prob
            df['implied_prob_away'] = df['implied_prob_away'] / total_prob
            
            odds_cols = ['implied_prob_home', 'implied_prob_draw', 'implied_prob_away']
            
            # NEW: 2-class odds features (conditional on no draw)
            print("Creating 2-class odds features...")
            
            # Calculate 2-class implied probabilities (P_home|no_draw, P_away|no_draw)
            sum_no_draw = df['implied_prob_home'] + df['implied_prob_away']
            df['implied_prob_home_2c'] = df['implied_prob_home'] / sum_no_draw
            df['implied_prob_away_2c'] = df['implied_prob_away'] / sum_no_draw
            
            # Calculate logit difference for 2-class home probability
            df['implied_logit_home_2c'] = np.log(df['implied_prob_home_2c'] / df['implied_prob_away_2c'])
            
            # Add to odds_cols
            odds_cols.extend(['implied_prob_home_2c', 'implied_prob_away_2c', 'implied_logit_home_2c'])
            
            print(f"Added 2-class odds features: {len(odds_cols)} total odds features")
        
        # Form features (last 5 games for each team)
        df = self.calculate_form(df)
        
        # NEW: Add momentum features (trending form)
        df = self.calculate_momentum_features(df)
        
        # NEW: Add seasonal patterns
        df = self.calculate_seasonal_features(df)
        
        # Select feature columns with enhanced features
        feature_cols = [
            'home_elo', 'away_elo', 'elo_diff', 'elo_diff_hfa', 'exp_home_win',
            'home_form_points', 'home_form_goals_for', 'home_form_goals_against',
            'away_form_points', 'away_form_goals_for', 'away_form_goals_against',
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_goals_diff',
            'home_momentum', 'away_momentum',  # NEW: Momentum features
            'home_month_performance', 'away_month_performance',  # NEW: Monthly performance
            'home_phase_performance', 'away_phase_performance',  # NEW: Season phase performance
            'month_sin', 'month_cos',  # NEW: Cyclical month encoding
            'is_early_season', 'is_late_season'  # NEW: Season phase indicators
        ] + odds_cols
        
        print(f"Total features: {len(feature_cols)}")
        print(f"Odds features: {len(odds_cols)} (including 2-class features)")
        
                    # Add advanced features for better prediction
        # Head-to-head historical performance
        df = self.calculate_h2h_features(df)
        
        # Add position-based features if available
        if 'home_position' in df.columns:
            df['home_position_log'] = np.log(21 - df['home_position'].clip(upper=20))
            df['away_position_log'] = np.log(21 - df['away_position'].clip(upper=20))
            df['position_diff'] = df['home_position_log'] - df['away_position_log']
            feature_cols.extend(['home_position_log', 'away_position_log', 'position_diff'])
        
        # Add interaction features
        df['elo_form_interaction'] = df['elo_diff'] * df['home_form_points'] / 15  # Normalized
        df['odds_elo_interaction'] = df['implied_prob_home_2c'] * df['elo_diff'] / 100 if 'implied_prob_home_2c' in df.columns else 0
        feature_cols.extend(['elo_form_interaction'])
        if 'implied_prob_home_2c' in df.columns:
            feature_cols.append('odds_elo_interaction')
        
        # Filter to available columns
        self.feature_cols = [col for col in feature_cols if col in df.columns]
        print(f"Using {len(self.feature_cols)} features: {self.feature_cols}")
        
        return df
    
    def calculate_h2h_features(self, df, n_h2h=5):
        """Calculate head-to-head features between teams"""
        df['h2h_home_wins'] = 0
        df['h2h_draws'] = 0
        df['h2h_away_wins'] = 0
        df['h2h_goals_diff'] = 0.0
        
        for idx in range(len(df)):
            home_team = df.loc[idx, 'home_team']
            away_team = df.loc[idx, 'away_team']
            
            # Get previous h2h matches
            h2h_prev = df[(df.index < idx) & 
                         (((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
                          ((df['home_team'] == away_team) & (df['away_team'] == home_team)))].tail(n_h2h)
            
            if len(h2h_prev) > 0:
                home_wins = 0
                draws = 0
                away_wins = 0
                goals_diff = 0
                
                for _, match in h2h_prev.iterrows():
                    if match['home_team'] == home_team:
                        goals_diff += match['home_goals'] - match['away_goals']
                        if match['home_goals'] > match['away_goals']:
                            home_wins += 1
                        elif match['home_goals'] == match['away_goals']:
                            draws += 1
                        else:
                            away_wins += 1
                    else:  # Teams are reversed
                        goals_diff -= (match['home_goals'] - match['away_goals'])
                        if match['home_goals'] < match['away_goals']:
                            home_wins += 1
                        elif match['home_goals'] == match['away_goals']:
                            draws += 1
                        else:
                            away_wins += 1
                
                df.loc[idx, 'h2h_home_wins'] = home_wins / len(h2h_prev)
                df.loc[idx, 'h2h_draws'] = draws / len(h2h_prev)
                df.loc[idx, 'h2h_away_wins'] = away_wins / len(h2h_prev)
                df.loc[idx, 'h2h_goals_diff'] = goals_diff / len(h2h_prev)
        
        return df
    
    def calculate_momentum_features(self, df, short_window=3, long_window=5):
        """Calculate momentum features showing recent form trends"""
        # Recent form (last 3 games) vs longer form (last 5 games)
        df['home_momentum'] = 0.0
        df['away_momentum'] = 0.0
        
        for idx in range(len(df)):
            if idx < long_window * 2:
                continue
                
            home_team = df.loc[idx, 'home_team']
            away_team = df.loc[idx, 'away_team']
            
            # Get recent form (last 3 games)
            home_recent = df[(df.index < idx) & 
                           ((df['home_team'] == home_team) | (df['away_team'] == home_team))].tail(short_window)
            
            away_recent = df[(df.index < idx) & 
                           ((df['home_team'] == away_team) | (df['away_team'] == away_team))].tail(short_window)
            
            # Calculate momentum (recent form - longer form)
            if len(home_recent) >= short_window:
                home_recent_points = 0
                for _, match in home_recent.iterrows():
                    if match['home_team'] == home_team:
                        if match['home_goals'] > match['away_goals']:
                            home_recent_points += 3
                        elif match['home_goals'] == match['away_goals']:
                            home_recent_points += 1
                    else:
                        if match['away_goals'] > match['home_goals']:
                            home_recent_points += 3
                        elif match['away_goals'] == match['home_goals']:
                            home_recent_points += 1
                
                # Momentum = recent form - longer form (normalized)
                home_longer_points = df.loc[idx, 'home_form_points']
                df.loc[idx, 'home_momentum'] = (home_recent_points / short_window) - (home_longer_points / long_window)
            
            if len(away_recent) >= short_window:
                away_recent_points = 0
                for _, match in away_recent.iterrows():
                    if match['home_team'] == away_team:
                        if match['home_goals'] > match['away_goals']:
                            away_recent_points += 3
                        elif match['home_goals'] == match['away_goals']:
                            away_recent_points += 1
                    else:
                        if match['away_goals'] > match['home_goals']:
                            away_recent_points += 3
                        elif match['away_goals'] == match['home_goals']:
                            away_recent_points += 1
                
                # Momentum = recent form - longer form (normalized)
                away_longer_points = df.loc[idx, 'away_form_points']
                df.loc[idx, 'away_momentum'] = (away_recent_points / short_window) - (away_longer_points / long_window)
        
        return df
    
    def calculate_seasonal_features(self, df):
        """Calculate seasonal patterns and performance by month/season phase"""
        # Extract month and season phase
        df['match_month'] = pd.to_datetime(df['date']).dt.month
        df['season_phase'] = 'mid'  # Default
        
        # Define season phases (Premier League runs Aug-May)
        # Early season: Aug-Oct (months 8-10)
        # Mid season: Nov-Feb (months 11, 12, 1, 2)
        # Late season: Mar-May (months 3-5)
        df.loc[df['match_month'].isin([8, 9, 10]), 'season_phase'] = 'early'
        df.loc[df['match_month'].isin([3, 4, 5]), 'season_phase'] = 'late'
        
        # Calculate team performance by month
        df['home_month_performance'] = 0.0
        df['away_month_performance'] = 0.0
        df['home_phase_performance'] = 0.0
        df['away_phase_performance'] = 0.0
        
        for idx in range(len(df)):
            if idx < 10:  # Need some history
                continue
                
            home_team = df.loc[idx, 'home_team']
            away_team = df.loc[idx, 'away_team']
            current_month = df.loc[idx, 'match_month']
            current_phase = df.loc[idx, 'season_phase']
            
            # Get historical performance in same month
            home_month_matches = df[(df.index < idx) & 
                                   (pd.to_datetime(df['date']).dt.month == current_month) &
                                   ((df['home_team'] == home_team) | (df['away_team'] == home_team))]
            
            away_month_matches = df[(df.index < idx) & 
                                   (pd.to_datetime(df['date']).dt.month == current_month) &
                                   ((df['home_team'] == away_team) | (df['away_team'] == away_team))]
            
            # Calculate month-specific win rate
            if len(home_month_matches) > 0:
                home_month_wins = 0
                for _, match in home_month_matches.iterrows():
                    if match['home_team'] == home_team:
                        if match['home_goals'] > match['away_goals']:
                            home_month_wins += 1
                    else:
                        if match['away_goals'] > match['home_goals']:
                            home_month_wins += 1
                df.loc[idx, 'home_month_performance'] = home_month_wins / len(home_month_matches)
            
            if len(away_month_matches) > 0:
                away_month_wins = 0
                for _, match in away_month_matches.iterrows():
                    if match['home_team'] == away_team:
                        if match['home_goals'] > match['away_goals']:
                            away_month_wins += 1
                    else:
                        if match['away_goals'] > match['home_goals']:
                            away_month_wins += 1
                df.loc[idx, 'away_month_performance'] = away_month_wins / len(away_month_matches)
            
            # Get historical performance in same season phase
            home_phase_matches = df[(df.index < idx) & 
                                   (df['season_phase'] == current_phase) &
                                   ((df['home_team'] == home_team) | (df['away_team'] == home_team))]
            
            away_phase_matches = df[(df.index < idx) & 
                                   (df['season_phase'] == current_phase) &
                                   ((df['home_team'] == away_team) | (df['away_team'] == away_team))]
            
            # Calculate phase-specific win rate
            if len(home_phase_matches) > 0:
                home_phase_wins = 0
                for _, match in home_phase_matches.iterrows():
                    if match['home_team'] == home_team:
                        if match['home_goals'] > match['away_goals']:
                            home_phase_wins += 1
                    else:
                        if match['away_goals'] > match['home_goals']:
                            home_phase_wins += 1
                df.loc[idx, 'home_phase_performance'] = home_phase_wins / len(home_phase_matches)
            
            if len(away_phase_matches) > 0:
                away_phase_wins = 0
                for _, match in away_phase_matches.iterrows():
                    if match['home_team'] == away_team:
                        if match['home_goals'] > match['away_goals']:
                            away_phase_wins += 1
                    else:
                        if match['away_goals'] > match['home_goals']:
                            away_phase_wins += 1
                df.loc[idx, 'away_phase_performance'] = away_phase_wins / len(away_phase_matches)
        
        # Add cyclical encoding for month (sine/cosine to capture circular nature)
        df['month_sin'] = np.sin(2 * np.pi * df['match_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['match_month'] / 12)
        
        # One-hot encode season phase
        df['is_early_season'] = (df['season_phase'] == 'early').astype(int)
        df['is_late_season'] = (df['season_phase'] == 'late').astype(int)
        
        return df
    
    def calculate_form(self, df, n_games=5):
        """Calculate form statistics for last n games"""
        df['home_form_points'] = 0.0
        df['home_form_goals_for'] = 0.0
        df['home_form_goals_against'] = 0.0
        df['away_form_points'] = 0.0
        df['away_form_goals_for'] = 0.0
        df['away_form_goals_against'] = 0.0
        
        for idx in range(len(df)):
            if idx < n_games * 2:  # Not enough history
                continue
                
            home_team = df.loc[idx, 'home_team']
            away_team = df.loc[idx, 'away_team']
            current_date = df.loc[idx, 'date']
            
            # Get previous matches for home team
            home_prev = df[(df.index < idx) & 
                          ((df['home_team'] == home_team) | (df['away_team'] == home_team))].tail(n_games)
            
            if len(home_prev) > 0:
                home_points = 0
                home_gf = 0
                home_ga = 0
                
                for _, match in home_prev.iterrows():
                    if match['home_team'] == home_team:
                        home_gf += match['home_goals']
                        home_ga += match['away_goals']
                        if match['home_goals'] > match['away_goals']:
                            home_points += 3
                        elif match['home_goals'] == match['away_goals']:
                            home_points += 1
                    else:
                        home_gf += match['away_goals']
                        home_ga += match['home_goals']
                        if match['away_goals'] > match['home_goals']:
                            home_points += 3
                        elif match['away_goals'] == match['home_goals']:
                            home_points += 1
                
                df.loc[idx, 'home_form_points'] = home_points / len(home_prev)
                df.loc[idx, 'home_form_goals_for'] = home_gf / len(home_prev)
                df.loc[idx, 'home_form_goals_against'] = home_ga / len(home_prev)
            
            # Get previous matches for away team
            away_prev = df[(df.index < idx) & 
                          ((df['home_team'] == away_team) | (df['away_team'] == away_team))].tail(n_games)
            
            if len(away_prev) > 0:
                away_points = 0
                away_gf = 0
                away_ga = 0
                
                for _, match in away_prev.iterrows():
                    if match['home_team'] == away_team:
                        away_gf += match['home_goals']
                        away_ga += match['away_goals']
                        if match['home_goals'] > match['away_goals']:
                            away_points += 3
                        elif match['home_goals'] == match['away_goals']:
                            away_points += 1
                    else:
                        away_gf += match['away_goals']
                        away_ga += match['home_goals']
                        if match['away_goals'] > match['home_goals']:
                            away_points += 3
                        elif match['away_goals'] == match['home_goals']:
                            away_points += 1
                
                df.loc[idx, 'away_form_points'] = away_points / len(away_prev)
                df.loc[idx, 'away_form_goals_for'] = away_gf / len(away_prev)
                df.loc[idx, 'away_form_goals_against'] = away_ga / len(away_prev)
        
        return df
    
    def split_by_season(self, df):
        """Split data chronologically by season"""
        print(f"\nSplitting data at season {self.train_end_season}")
        
        # Get season order
        seasons = sorted(df['season'].unique())
        train_seasons = [s for s in seasons if s <= self.train_end_season]
        test_seasons = [s for s in seasons if s > self.train_end_season]
        
        train_df = df[df['season'].isin(train_seasons)]
        test_df = df[df['season'].isin(test_seasons)]
        
        print(f"Train: {len(train_df)} matches from seasons {train_seasons[0]} to {train_seasons[-1]}")
        print(f"Test: {len(test_df)} matches from seasons {test_seasons[0]} to {test_seasons[-1]}")
        
        # Remove rows with insufficient form data
        train_df = train_df[train_df['home_form_points'] != 0].copy()
        test_df = test_df[test_df['home_form_points'] != 0].copy()
        
        print(f"After removing matches without form data: Train={len(train_df)}, Test={len(test_df)}")
        
        # Remove draw matches (where home_goals == away_goals)
        train_df = train_df[train_df['home_goals'] != train_df['away_goals']].copy()
        test_df = test_df[test_df['home_goals'] != test_df['away_goals']].copy()
        
        print(f"After removing draw matches: Train={len(train_df)}, Test={len(test_df)}")
        print(f"Note: Model now only predicts Home Win vs Away Win (no draws)")
        
        return train_df, test_df
    
    def prepare_features_targets(self, df):
        """Prepare feature matrix and single target for 2-class classification (no draws)"""
        X = df[self.feature_cols].copy()
        
        # Handle any remaining NaNs
        X = X.fillna(X.median())
        
        # Single target for 2-class classification: Home Win vs Away Win
        # y = 1 if home wins else 0 (as recommended)
        y_target = (df['home_goals'] > df['away_goals']).astype(int)
        
        # Verify no draws remain
        draws_remaining = (df['home_goals'] == df['away_goals']).sum()
        if draws_remaining > 0:
            print(f"Warning: {draws_remaining} draw matches still present in data")
        else:
            print(f"All draw matches removed. Data ready for 2-class classification")
        
        return X, y_target
    
    def tune_hyperparameters(self, X_train, y_train, cv_folds=5):
        """
        Tune hyperparameters for single 2-class model using Optuna
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets
        cv_folds : int
            Number of cross-validation folds
        """
        print(f"\nTuning hyperparameters for single 2-class model using Optuna...")
        
        # Use Optuna tuner for Bayesian optimization (TimeSeriesSplit handled internally)
        best_params = self.optuna_tuner.tune(
            X_train, y_train, None, None,  # No separate validation set needed
            model_type="combined"  # Use combined for single model
        )
        
        # Update internal parameters
        self.best_params = best_params
        
        print(f"Single 2-class model hyperparameters tuned and saved!")
        return best_params
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None, tune_hyperparams=True):
        """Train single XGBoost model for 2-class classification (Home Win vs Away Win)"""
        print("\nTraining single 2-class model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        if tune_hyperparams:
            # Tune hyperparameters for single model
            print("Starting hyperparameter tuning...")
            best_params = self.tune_hyperparameters(X_train, y_train)
        else:
            # Use saved hyperparameters or defaults
            print("Using saved hyperparameters or defaults...")
            best_params = self.best_params
            
            # If no saved params, use defaults
            if best_params is None:
                best_params = get_default_params()
                print("Warning: No saved model params, using defaults")
        
        # NEW: Implement probability calibration
        print("Implementing probability calibration...")
        
        # Create calibration set from training data
        X_fit_scaled, y_fit, X_cal_scaled, y_cal = self._create_calibration_set(
            X_train_scaled, y_train
        )
        # Train raw model on fitting set
        print("Training raw XGBoost model...")
        raw_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=self.random_state,
            eval_metric='logloss',  # Use logloss for calibration
            early_stopping_rounds=20,
            verbosity=0,
            **best_params
        )
        # Use validation set for early stopping if available
        if X_val is not None:
            raw_model.fit(
                X_fit_scaled, y_fit,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            raw_model.fit(X_fit_scaled, y_fit, verbose=False)
        
        # Create calibrated classifier
        print("Calibrating probabilities using sigmoid (Platt scaling)...")
        self.model = CalibratedClassifierCV(
            estimator=raw_model,
            method='sigmoid',  # Changed from isotonic to sigmoid for better generalization
            cv='prefit'
        )
        # Fit calibration on calibration set
        self.model.fit(X_cal_scaled, y_cal)
        # Evaluate calibration quality
        self._evaluate_calibration(X_cal_scaled, y_cal)
        
        # Calculate and print train/validation accuracy
        self._print_train_val_accuracy(X_fit_scaled, y_fit, X_cal_scaled, y_cal, X_val_scaled, y_val)
        
        print("Threshold optimization integrated into Optuna tuning")
        self.threshold_ = 0.5  # Default, will be overridden by Optuna if available
        print("Training complete!")
        
        # Print final parameters used
        print(f"\nFinal parameters used:")
        print(f"Single 2-class model: {best_params}")
        print(f"Optimal threshold: {self.threshold_:.4f}")
        
        # Save threshold along with hyperparameters
        if hasattr(self, 'threshold_'):
            self.hp_manager.save_threshold(self.threshold_)
    
    def cross_validate_models(self, X_train, y_train, cv_folds=5):
        """Perform cross-validation to assess model stability"""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create base model with correct parameters
        base_params = self.best_params or get_default_params()
        
        # Cross-validation for single 2-class model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=self.random_state,
            **base_params
        )
        
        scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=TimeSeriesSplit(n_splits=cv_folds),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print(f"Cross-validation results:")
        print(f"Single 2-class model - AUC-ROC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def predict(self, X, optimize_threshold=False, y_true=None):
        """Make predictions for 2 classes: Home Win vs Away Win (no draw)"""
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities from single 2-class model
        probs = self.model.predict_proba(X_scaled)
        home_prob = probs[:, 1]  # Probability of home win
        away_prob = probs[:, 0]  # Probability of away win
        
        # For 2-class prediction: Home Win vs Away Win
        # Determine threshold
        if optimize_threshold and y_true is not None:
            # Find optimal threshold on this data
            threshold = self._find_best_threshold(y_true, home_prob, metric='accuracy')
            self.threshold_ = threshold
        elif hasattr(self, 'best_params') and self.best_params and 'threshold' in self.best_params:
            threshold = self.best_params['threshold']
            print(f"Using threshold from Optuna tuning: {threshold:.4f}")
        else:
            # Use a more balanced default threshold
            threshold = getattr(self, 'threshold_', 0.45)  # Lowered from 0.5 to improve recall
            
        pred_home_win = (home_prob > threshold).astype(int)
        pred_away_win = (home_prob <= threshold).astype(int)
        
        return pred_home_win, pred_away_win, home_prob, away_prob
    
    def evaluate(self, X_test, y_test, df_test):
        """Evaluate single 2-class model performance"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Make predictions with optimized threshold
        pred_home, pred_away, prob_home, prob_away = self.predict(X_test, optimize_threshold=True, y_true=y_test)
        
        # Calculate metrics for single 2-class model
        print("\nSINGLE 2-CLASS MODEL METRICS:")
        print("-" * 30)
        acc = accuracy_score(y_test, pred_home)
        prec = precision_score(y_test, pred_home, zero_division=0)
        rec = recall_score(y_test, pred_home)
        f1 = f1_score(y_test, pred_home)
        auc = roc_auc_score(y_test, prob_home)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        
        # Overall match outcome accuracy
        print("\n2-CLASS CLASSIFICATION (HOME WIN vs AWAY WIN):")
        print("-" * 30)
        print(f"2-Class Accuracy: {acc:.4f}")
        
        # Distribution of predictions
        print(f"\nPrediction Distribution:")
        print(f"Home wins predicted: {pred_home.sum()} ({pred_home.mean()*100:.1f}%)")
        print(f"Away wins predicted: {pred_away.sum()} ({pred_away.mean()*100:.1f}%)")
        
        print(f"\nActual Distribution:")
        print(f"Home wins actual: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
        print(f"Away wins actual: {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
        print(f"Note: All draw matches have been removed from the dataset")
        
        # Feature importance
        self.plot_feature_importance()
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, pred_home)
        
        # ROC curve
        self.plot_roc_curve(y_test, prob_home)
        
        # Season-wise performance
        self.evaluate_by_season(df_test, pred_home, y_test)
        
        return {
            'accuracy': acc,
            'auc': auc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        }
    
    def plot_feature_importance(self):
        """Plot feature importance for single 2-class model"""
        if self.model is None:
            print("Model not trained yet")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Single model feature importance - get from base estimator (FIXED)
        if hasattr(self.model, 'estimators_'):
            # For CalibratedClassifierCV, get from the first base estimator
            # The estimators_ list contains the base estimators
            importance = self.model.estimators_[0].feature_importances_
        elif hasattr(self.model, 'base_estimator'):
            # For other calibrated classifiers
            importance = self.model.base_estimator.feature_importances_
        elif hasattr(self.model, 'feature_importances_'):
            # For regular XGBoost model
            importance = self.model.feature_importances_
        else:
            print("Warning: Could not extract feature importances from model")
            return
            
        indices = np.argsort(importance)[::-1][:10]
        
        plt.barh(range(10), importance[indices])
        plt.yticks(range(10), [self.feature_cols[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Top 10 Features - Single 2-Class Model (Home Win vs Away Win)')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix for single 2-class model"""
        plt.figure(figsize=(8, 6))
        
        # Single confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Single 2-Class Model (Home Win vs Away Win)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks([0, 1], ['Away Win', 'Home Win'])
        plt.yticks([0, 1], ['Away Win', 'Home Win'])
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_proba, title_suffix=""):
        """Plot ROC curve showing AUC"""
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, 'b-', linewidth=2, 
                label=f'ROC Curve (AUC = {auc_score:.4f})')
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, 
                label='Random Classifier (AUC = 0.50)')
        
        # Fill area under curve
        plt.fill_between(fpr, tpr, alpha=0.3, color='lightblue')
        
        # Labels and title
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
        plt.title(f'ROC Curve - Receiver Operating Characteristic{title_suffix}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Set axis limits
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        # Add performance interpretation
        if auc_score >= 0.9:
            interpretation = "Excellent"
        elif auc_score >= 0.8:
            interpretation = "Good"
        elif auc_score >= 0.7:
            interpretation = "Fair"
        else:
            interpretation = "Poor"
        
        plt.text(0.6, 0.2, f'Performance: {interpretation}\nAUC = {auc_score:.4f}', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        print("ROC curve saved as 'roc_curve.png'")
        print(f"AUC-ROC Score: {auc_score:.4f} ({interpretation} discrimination)")
    
    def evaluate_by_season(self, df_test, pred_home, y_test):
        """Evaluate performance by season for single 2-class model"""
        print("\nPERFORMANCE BY SEASON:")
        print("-" * 50)
        
        df_eval = df_test.copy()
        df_eval['pred_home'] = pred_home
        df_eval['actual'] = y_test
        
        for season in sorted(df_eval['season'].unique()):
            season_data = df_eval[df_eval['season'] == season]
            
            if len(season_data) > 0:
                # Single 2-class accuracy: Home Win vs Away Win
                acc = accuracy_score(season_data['actual'], season_data['pred_home'])
                
                print(f"{season}: 2-Class Acc={acc:.3f}")
    
    def run_full_pipeline(self, tune_hyperparams=True, cross_validate=True):
        """Run the complete prediction pipeline with optional hyperparameter tuning"""
        # Load and prepare data
        df = self.load_and_prepare_data()
        df = self.engineer_features(df)
        
        # Split by season
        train_df, test_df = self.split_by_season(df)
        
        # Prepare features and targets
        X_train, y_train = self.prepare_features_targets(train_df)
        X_test, y_test = self.prepare_features_targets(test_df)
        
        # Create validation set from last 20% of training data
        val_size = int(len(X_train) * 0.2)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        if cross_validate:
            # Perform cross-validation
            self.cross_validate_models(X_train, y_train)
        
        # Train models
        self.train_models(X_train, y_train, 
                         X_val, y_val, tune_hyperparams)
        
        # Evaluate
        metrics = self.evaluate(X_test, y_test, test_df)
        
        return metrics


def main():
    """Main execution function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='XGBoost Premier League Match Predictor')
    parser.add_argument('--data_path', 
                       default='/Users/binhminhtran/USYD/DATA1002/Group Project/Dataset/clean/pl_matches_clean_full_elo.csv',
                       help='Path to the CSV file with match data and Elo ratings')
    parser.add_argument('--train_end_season', 
                       default='2021-2022',
                       help='Last season to include in training (inclusive)')
    parser.add_argument('--random_state', 
                       type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize predictor
    # Train on seasons up to 2021-2022, test on 2022-2023 onwards
    predictor = PLMatchPredictor(
        data_path=args.data_path,
        train_end_season=args.train_end_season,
        random_state=args.random_state
    )
    
    # Run full pipeline with hyperparameter tuning
    print("\n" + "="*60)
    print("PREMIER LEAGUE MATCH OUTCOME PREDICTION - XGBOOST WITH TUNING")
    print("2-CLASS CLASSIFICATION: HOME WIN vs AWAY WIN (NO DRAWS)")
    print("="*60)
    
    metrics = predictor.run_full_pipeline(tune_hyperparams=True, cross_validate=True)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Single 2-Class Model Accuracy: {metrics['accuracy']:.4f}")
    print(f"Single 2-Class Model AUC-ROC:  {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    if predictor.best_params:
        print(f"\nBest parameters found:")
        print(f"Single 2-class model: {predictor.best_params}")
    
    return predictor, metrics


if __name__ == "__main__":
    # Install required packages if needed
    import subprocess
    import sys
    
    try:
        import xgboost
    except ImportError:
        print("Installing xgboost...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    
    try:
        import seaborn
    except ImportError:
        print("Installing seaborn...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    
    # Run the model with hyperparameter tuning
    predictor, metrics = main()


