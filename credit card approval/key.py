import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
from imblearn.over_sampling import SMOTE  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier  
from xgboost import XGBClassifier  
from matplotlib.backends.backend_pdf import PdfPages
import io

# Step 1: Load Dataset  
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'

# Load dataset with proper column names  
data = pd.read_csv(url, names=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"], header=None)  

# Display first few rows of the dataset  
print("First 5 rows of dataset:")  
print(data.head())  

# Step 2: Data Preprocessing  
data.replace('?', np.nan, inplace=True)  

# Convert numerical columns  
numerical_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']  
for col in numerical_cols:  
    data[col] = pd.to_numeric(data[col], errors='coerce')  
    data[col] = data[col].fillna(data[col].median())  # Fixed inplace warning

# Fill missing values in categorical columns  
categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']  
for col in categorical_cols:  
    data[col] = data[col].fillna(data[col].mode()[0])  # Fixed inplace warning

# Ensure no missing values in target column  
print(f"Missing values in A16: {data['A16'].isnull().sum()}")  
data = data.dropna(subset=['A16'])  # Fixed inplace warning

# Encode categorical columns  
label_encoders = {}  
for col in categorical_cols:  
    le = LabelEncoder()  
    data[col] = le.fit_transform(data[col])  
    label_encoders[col] = le  

# Feature scaling  
scaler = StandardScaler()  
X = pd.DataFrame(scaler.fit_transform(data.drop('A16', axis=1)), columns=data.columns[:-1])  

# Encode target variable  
y = data['A16'].apply(lambda x: 1 if x == '+' else 0)  

# Step 3: Train-Test Split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  

# Step 4: Handle Class Imbalance using SMOTE  
print("Class distribution before SMOTE:", y_train.value_counts())  
smote = SMOTE(sampling_strategy='auto', random_state=42)  
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)  
print("Class distribution after SMOTE:", pd.Series(y_train_resampled).value_counts())  

# Step 5: Model Training and Evaluation  
models = {  
    'Logistic Regression': LogisticRegression(),  
    'Decision Tree': DecisionTreeClassifier(),  
    'Random Forest': RandomForestClassifier(),  
    'Gradient Boosting': GradientBoostingClassifier(),  
    'XGBoost': XGBClassifier(use_label_encoder=False)  
}  

# Create a PdfPages object to store results
with PdfPages('model_evaluation_results.pdf') as pdf:

    def evaluate_model(model, X_train, X_test, y_train, y_test):  
        model.fit(X_train, y_train)  
        y_pred = model.predict(X_test)  
        accuracy = accuracy_score(y_test, y_pred)  

        # Print and save accuracy
        print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}")  
        output = io.StringIO()  # Capture output in string
        output.write(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}\n")

        # Classification report with better alignment
        output.write("\nClassification Report:\n")
        report = classification_report(y_test, y_pred)
        output.write(report)

        # Confusion matrix
        output.write("\nConfusion Matrix:\n")
        cm = confusion_matrix(y_test, y_pred)
        output.write(str(cm))

        # Save text to PDF
        plt.figure(figsize=(8, 6))
        plt.text(0.1, 0.9, output.getvalue(), fontsize=10, ha='left', va='top', wrap=True)
        plt.axis('off')  # Turn off axes
        pdf.savefig()  # Save text to PDF
        plt.close()

        # Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Approved', 'Approved'], yticklabels=['Not Approved', 'Approved'])
        ax.set_title(f'{model.__class__.__name__} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        pdf.savefig(fig)  # Save plot to PDF
        plt.close()

    for name, model in models.items():  
        print(f"Evaluating {name}...")  
        evaluate_model(model, X_train_resampled, X_test, y_train_resampled, y_test)  

    # Store the trained Random Forest model for prediction
    rf_model = models['Random Forest']

    # Interactive Prediction
    def predict_credit_card_approval(model, input_data):  
        if len(input_data) != X.shape[1]:  
            raise ValueError(f"Expected {X.shape[1]} features, but got {len(input_data)}")  

        input_data = np.array(input_data).reshape(1, -1)  
        input_data_scaled = scaler.transform(input_data)  
        prediction = model.predict(input_data_scaled)[0]  
        return "Approved" if prediction == 1 else "Not Approved"  

    # Example Applicant Data  
    new_applicant = [0, 30.0, 2.5, 1, 1, 0, 2, 50000.0, 1, 0, 200, 0, 1, 0, 100]  

    # Check feature count  
    print(f"Expected Features: {X.shape[1]}, Provided Features: {len(new_applicant)}")  

    try:  
        prediction = predict_credit_card_approval(rf_model, new_applicant)  
        print(f"Prediction for new applicant: {prediction}")  
    except ValueError as e:  
        print(f"Error: {e}")
