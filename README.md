# CS2 Match Data Analysis and Prediction Project

A comprehensive machine learning project for analyzing and predicting Counter-Strike 2 (CS2) professional match outcomes using map-level data, team statistics, and advanced modeling techniques.

## Project Overview

This project implements a complete pipeline for CS2 match prediction, from data collection to deployment-ready models. It includes data scraping, feature engineering, multiple machine learning approaches, and comprehensive evaluation frameworks.

## Project Structure

### Dataset: Data Collection and Preprocessing
- **Data Scraping**: Automated collection from BO3.gg API
- **Feature Engineering**: Chronological winrates, recent form, and contextual features
- **Dataset Extension**: Automated updates with new tournament data

### Evaluation: Machine Learning Models
- **Classification**: Binary win/loss prediction with multiple algorithms
- **Regression**: Probabilistic win prediction with calibration

### Phase3: Analysis and Reporting
- **Model Comparison**: Comprehensive evaluation across multiple metrics
- **Calibration Analysis**: Probability reliability assessment
- **Final Report**: Complete scientific analysis and findings

## Key Features

🎯 **Comprehensive ML Pipeline**
- Multiple algorithms: Logistic Regression, Random Forest, XGBoost, LightGBM, MLP
- Advanced preprocessing with temporal awareness to prevent data leakage
- Automated hyperparameter tuning and model selection

📊 **Rich Feature Engineering**
- Chronological team performance metrics
- Map-specific winrates and preferences
- Head-to-head statistics and recent form
- Pick/ban context and strategic indicators

🔄 **Automated Data Management**
- Real-time tournament detection and dataset updates
- Duplicate prevention and data quality controls
- Backup systems and error handling

📈 **Advanced Evaluation**
- Multiple evaluation metrics (ROC-AUC, Brier Score, Log-Loss)
- Calibration analysis and reliability assessment
- Per-map model specialization and routing

## Quick Start

### Prerequisites
```bash
pip install -r Dataset/requirements.txt
```

### Data Collection
```bash
cd Dataset
python cs2_match_data_scraper.py      # Collect raw match data
python create_ml_dataset.py           # Create ML-ready dataset
python extend_dataset.py              # Update with new tournaments (optional)
```

### Model Training
```bash
cd Evaluation/Classification
# Open and run cs2_map_prediction.ipynb for classification models

cd ../Regression  
# Open and run cs2_map_probability_regression.ipynb for regression models
```

## Dataset

- **Source**: Professional CS2 matches from BO3.gg API
- **Timeframe**: Last 6 months of tournament data
- **Scale**: 2,400+ map records from 1,000+ matches
- **Teams**: Top 50 teams by earnings
- **Features**: 20+ engineered features including temporal dynamics

## Model Performance

| Model | ROC-AUC | Brier Score | Log-Loss | Accuracy |
|-------|---------|-------------|----------|----------|
| LightGBM | 0.72 | 0.231 | 0.659 | 67.2% |
| Random Forest | 0.71 | 0.233 | 0.665 | 66.8% |
| XGBoost | 0.70 | 0.235 | 0.672 | 66.1% |
| MLP | 0.69 | 0.238 | 0.675 | 65.7% |

## Repository Structure

```
├── Dataset/                         # Data collection and preprocessing
│   ├── cs2_match_data_scraper.py   # Main data scraper
│   ├── create_ml_dataset.py        # ML dataset transformation
│   ├── extend_dataset.py           # Automated dataset updates
│   ├── dataset_profiler.py         # Data quality analysis
│   └── cs2api/                     # Custom API library
├── Evaluation/                      # Machine learning models
│   ├── Classification/             # Classification models and analysis
│   ├── Regression/                 # Regression models and calibration
│   └── compilated/                # Results aggregation and reporting
└── Phase3/                          # Final analysis and documentation
```

## Technical Highlights

- **Temporal Awareness**: All features computed chronologically to prevent data leakage
- **Advanced Calibration**: Probability reliability optimization for betting applications
- **Per-Map Specialization**: Custom models for different map types and contexts
- **Robust Evaluation**: Multiple train/validation/test strategies with proper grouping
- **Production Ready**: Automated pipelines with comprehensive error handling

## Contributing

This project was developed as part of a machine learning course focusing on real-world applications of predictive modeling in esports analytics.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- BO3.gg for providing the CS2 match data API
- The professional CS2 community for generating the rich dataset
- Various open-source libraries enabling this comprehensive analysis