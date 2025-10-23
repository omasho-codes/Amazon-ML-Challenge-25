# ==============================================================================
# 1. SETUP AND DATA PREPARATION
# ==============================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

# Suppress verbose warnings
warnings.filterwarnings('ignore')
print("✅ Libraries imported successfully.")

# --- Helper Functions ---
def calculate_smape(y_true, y_pred):
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = 2 * np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred)
    raw_values = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)
    return np.mean(raw_values) * 100

def lgbm_smape(y_true, y_pred):
    """Custom SMAPE metric for LightGBM, converting log values back to original scale."""
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    smape_score = calculate_smape(y_true_orig, y_pred_orig)
    return 'smape', smape_score, False # False means lower is better

# --- Load and Prepare Data ---
print("\n🔄 Loading and preparing data...")
try:
    df_train = pd.read_csv("../Mukil/train_embeddings.csv")
    df_test = pd.read_csv("../Mukil/test_embeddings.csv")
except FileNotFoundError:
    print("❌ Error: Make sure 'train_embeddings.csv' and 'test_embeddings.csv' are in the '../Mukil/' directory.")

X = df_train.drop(['price', 'sample_id'], axis=1)
y = df_train['price']
X_final_test = df_test.drop(['sample_id'], axis=1)

# Apply log1p transformation to the target variable
y_log = np.log1p(y)

# Split data into training and validation sets
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
y_test_orig = np.expm1(y_test_log) # Pre-calculate for evaluation

print(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} validation samples.")
print("-" * 60)


# ==============================================================================
# 2. FEATURE IMPORTANCE GENERATION (ON CPU)
# ==============================================================================
print("\n🔄 Generating feature importances for each model type on CPU...")

# --- XGBoost Feature Importances ---
initial_xgb = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, eval_metric='mae', early_stopping_rounds=50, random_state=42)
initial_xgb.fit(X_train, y_train_log, eval_set=[(X_test, y_test_log)], verbose=False)
xgb_importances = pd.DataFrame({'feature': X.columns, 'importance': initial_xgb.feature_importances_}).sort_values('importance', ascending=False)
print("✅ XGBoost feature importances calculated.")

# --- CatBoost Feature Importances ---
initial_cat = CatBoostRegressor(loss_function='MAE', iterations=1000, learning_rate=0.05, depth=5, subsample=0.8, colsample_bylevel=0.8, eval_metric='MAE', random_seed=42)
initial_cat.fit(X_train, y_train_log, eval_set=[(X_test, y_test_log)], early_stopping_rounds=100, verbose=False)
cat_importances = pd.DataFrame({'feature': X.columns, 'importance': initial_cat.get_feature_importance()}).sort_values('importance', ascending=False)
print("✅ CatBoost feature importances calculated.")

# --- LightGBM Feature Importances ---
initial_lgbm = lgb.LGBMRegressor(objective='mae', n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
initial_lgbm.fit(X_train, y_train_log, eval_set=[(X_test, y_test_log)], eval_metric=lgbm_smape, callbacks=[lgb.early_stopping(100, verbose=False)])
lgbm_importances = pd.DataFrame({'feature': X.columns, 'importance': initial_lgbm.feature_importances_}).sort_values('importance', ascending=False)
print("✅ LightGBM feature importances calculated.")
print("-" * 60)


# ==============================================================================
# 3. TRAINING & PREDICTION FOR ALL 12 MODELS (ON CPU)
# ==============================================================================
feature_counts = [500, 1000, 2000, 3500]
all_oof_preds = {} # To store out-of-fold (validation) predictions
all_final_preds = {} # To store final test set predictions

model_configs = {
    'XGBoost': {
        'model': xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=2000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, eval_metric='mae', early_stopping_rounds=50, random_state=42),
        'fit_params': {'verbose': False},
        'importances': xgb_importances
    },
    'CatBoost': {
        'model': CatBoostRegressor(loss_function='MAE', iterations=2000, learning_rate=0.05, depth=5, subsample=0.8, colsample_bylevel=0.8, eval_metric='MAE', random_seed=42),
        'fit_params': {'early_stopping_rounds': 100, 'verbose': False},
        'importances': cat_importances
    },
    'LightGBM': {
        'model': lgb.LGBMRegressor(objective='mae', n_estimators=2000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        'fit_params': {'eval_metric': lgbm_smape, 'callbacks': [lgb.early_stopping(100, verbose=False)]},
        'importances': lgbm_importances
    }
}

for model_name, config in model_configs.items():
    print(f"\n🚀 Starting training for {model_name} models...")
    oof_preds_list = []
    final_preds_list = []

    for n in feature_counts:
        print(f"  - Training with top {n} features...")

        # Select top N features
        top_n_features = config['importances']['feature'].head(n).tolist()
        X_train_subset = X_train[top_n_features]
        X_test_subset = X_test[top_n_features]
        X_final_test_subset = X_final_test[top_n_features]

        # Train the model
        model = config['model']
        model.fit(X_train_subset, y_train_log, eval_set=[(X_test_subset, y_test_log)], **config['fit_params'])

        # Store predictions
        oof_preds_list.append(model.predict(X_test_subset))
        final_preds_list.append(model.predict(X_final_test_subset))

    all_oof_preds[model_name] = np.mean(oof_preds_list, axis=0)
    all_final_preds[model_name] = np.mean(final_preds_list, axis=0)
    print(f"✅ {model_name} training and prediction complete.")
print("-" * 60)


# ==============================================================================
# 4. ENSEMBLE EVALUATION AND SUBMISSION
# ==============================================================================
print("\n📊 Evaluating ensembles and creating submission files...")

# --- Evaluate and save individual averaged models ---
for model_name in model_configs.keys():
    # Evaluate
    y_pred_log = all_oof_preds[model_name]
    y_pred_orig = np.expm1(y_pred_log)
    smape = calculate_smape(y_test_orig, y_pred_orig)
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    print(f"\n--- {model_name} Averaged Ensemble ---")
    print(f"SMAPE on Test Set: {smape:.4f}%")
    print(f"MSE on Test Set:   {mse:.4f}")

    # Create and save submission file
    final_preds_orig = np.expm1(all_final_preds[model_name])
    submission_df = pd.DataFrame({'sample_id': df_test['sample_id'], 'price': final_preds_orig})
    filename = f"averaged_{model_name.lower()}_submission.csv"
    submission_df.to_csv(filename, index=False)
    print(f"💾 Submission file saved as '{filename}'")

# --- Evaluate and save the Grand Ensemble ---
grand_ensemble_oof_preds = np.mean([all_oof_preds['XGBoost'], all_oof_preds['CatBoost'], all_oof_preds['LightGBM']], axis=0)
grand_ensemble_final_preds = np.mean([all_final_preds['XGBoost'], all_final_preds['CatBoost'], all_final_preds['LightGBM']], axis=0)

# Evaluate
y_pred_orig_grand = np.expm1(grand_ensemble_oof_preds)
smape_grand = calculate_smape(y_test_orig, y_pred_orig_grand)
mse_grand = mean_squared_error(y_test_orig, y_pred_orig_grand)
print("\n--- 🏆 Grand Ensemble (XGB+CAT+LGBM) ---")
print(f"SMAPE on Test Set: {smape_grand:.4f}%")
print(f"MSE on Test Set:   {mse_grand:.4f}")

# Create and save submission file
final_preds_orig_grand = np.expm1(grand_ensemble_final_preds)
submission_df_grand = pd.DataFrame({'sample_id': df_test['sample_id'], 'price': final_preds_orig_grand})
filename_grand = "grand_ensemble_submission.csv"
submission_df_grand.to_csv(filename_grand, index=False)
print(f"💾 Final submission file saved as '{filename_grand}'")
print("-" * 60)
print("\n🎉 All processes complete!")
