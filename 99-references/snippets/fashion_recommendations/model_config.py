"""
Model hyperparameters and configuration
"""

# Evaluation configuration
EVAL_CONFIG = {
    "k": 12,  # MAP@K
    "prediction_window_days": 7,
    "train_cutoff": "2020-09-01",
    "val_cutoff": "2020-09-15",
    "test_cutoff": "2020-09-22"
}

# Popularity model
POPULARITY_CONFIG = {
    "n_days": 7,
    "alpha": 0.5,  # recency weight decay
    "top_n": 12
}

# Age-based rules model
AGE_RULES_CONFIG = {
    "age_bins": [18, 25, 35, 45, 55, 100],
    "top_n_per_segment": 12
}

# Ensemble model
ENSEMBLE_CONFIG = {
    "models": ["popularity", "age_rules", "simple_mlp"],
    "weights": {
        "popularity": 0.2,
        "age_rules": 0.3,
        "simple_mlp": 0.5
    }
}
