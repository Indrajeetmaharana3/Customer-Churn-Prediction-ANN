# Customer Churn Prediction using ANN

A beginner-friendly Deep Learning binary-classification project built with TensorFlow/Keras. It predicts whether a customer may churn and includes a Streamlit interface.

**Dataset notice:** `data/customer_churn.csv` is synthetic data generated for learning. It is not real customer information. Do not present its metrics as real-world results.

## Requirements
- Python 3.11 or 3.12 recommended
- VS Code
- Internet connection to install dependencies

If Python 3.14 is your only installed version, install/use Python 3.12 for this TensorFlow project.

## Run in VS Code (Windows PowerShell)

1. Open this folder in VS Code.
2. Create a virtual environment:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If activation is blocked:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```
3. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Train and evaluate the ANN:
   ```powershell
   python train.py
   ```
   This saves the Keras model, preprocessing pipeline, evaluation metrics and training-loss plot.
5. Run the app:
   ```powershell
   streamlit run app.py
   ```

## Project workflow
1. Load and inspect the CSV with Pandas.
2. Separate features and target.
3. Split training and test data.
4. Scale numeric features and one-hot encode categorical features.
5. Build an ANN with Dense and Dropout layers.
6. Train with Adam, binary cross-entropy and early stopping.
7. Evaluate using accuracy, precision, recall, F1-score, ROC-AUC and confusion matrix.
8. Save the model and preprocessing pipeline.
9. Use Streamlit for interactive predictions.

## Resume description (use after successfully running and understanding the project)

**Customer Churn Prediction using ANN | Deep Learning**
- Built a feed-forward Artificial Neural Network using TensorFlow/Keras for binary customer churn classification.
- Preprocessed numerical and categorical features using StandardScaler and OneHotEncoder.
- Evaluated model performance using accuracy, precision, recall, F1-score and ROC-AUC.
- Saved the trained model and developed an interactive Streamlit prediction application.

**Technologies:** Python, TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Joblib, Streamlit.

Only add actual metric values after running the project and checking `models/metrics.json`.
