from flask import Flask, render_template
import pickle
import pandas as pd
import shap
import zipfile
import xgboost as xgb
import warnings

# Suppress the XGBoost warning
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')



app = Flask(__name__)

@app.route('/')
def index():
    # Unzip and load the second DataFrame
    with zipfile.ZipFile('X_encoded_subset.zip', 'r') as zipf2:
        with zipf2.open('X_test_encoded_subset.csv') as file2:
            encoded = pd.read_csv(file2)

    load_xgb = xgb.Booster(model_file='xgb_model.json')

    features = encoded.columns

    explainer = shap.TreeExplainer(load_xgb)

    selected_data = encoded.head(1)
    selected_data = selected_data.iloc[:, :]

    X_sample = selected_data.values.reshape(1, -1)
    shap_values = explainer.shap_values(X_sample)

    shap_df = pd.DataFrame(data=shap_values, columns=features)
    top_10_shap = shap_df.abs().stack().sort_values(ascending=False).head(10)

    return render_template('shap_results.html', top_10_shap=top_10_shap)

if __name__ == '__main__':
    app.run(debug=True)

