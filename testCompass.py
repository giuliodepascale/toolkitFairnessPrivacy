import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from privacy_fairness_toolkit import Toolkit

# Load COMPAS dataset
compas_data = pd.read_csv('compas-scores-raw.csv')

# Selecting relevant columns
compas_data_relevant = compas_data[['Sex_Code_Text', 'Ethnic_Code_Text', 'DecileScore', 'ScoreText', 'AssessmentType', 'RawScore']]
compas_data_relevant = compas_data_relevant.dropna()

# Define sensitive features
sensitive_features = ['Ethnic_Code_Text']

# Define target variable
target = 'ScoreText'
compas_data_relevant[target] = LabelEncoder().fit_transform(compas_data_relevant[target])

# Define features and encode categorical variables
X = compas_data_relevant.drop(columns=[target] + sensitive_features)
y = compas_data_relevant[target]
X = pd.get_dummies(X, drop_first=True)
X = StandardScaler().fit_transform(X)

sensitive_features_data = compas_data_relevant[sensitive_features]

# Dividi il dataset includendo le feature sensibili
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, sensitive_features_data, test_size=0.2, random_state=42
)

sensitive_feature = sensitive_test['Ethnic_Code_Text'].values

# Reset degli indici per allinearli


# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "XGBoost": XGBClassifier()

    
   
}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_test = y_test.reset_index(drop=True)
    predictions = pd.Series(predictions).reset_index(drop=True)
    # Initialize toolkit with predictions, labels, and sensitive features
    toolkit = Toolkit(predictions,y_test, sensitive_feature)
    print(model_name)
    print("modello senza privacy:")
    toolkit.summary_fairness_accuracy()
    # Evaluate trade-off between accuracy and fairness using the toolkit
    result = toolkit.evaluate_tradeoff_accuracy_fairness(noise_type='laplace', data_type='categorical',delta=0.1)
   

