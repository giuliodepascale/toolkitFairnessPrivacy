import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from privacy_fairness_toolkit import Toolkit

# Caricamento del dataset Adult Census
adult_data = pd.read_csv('adult.csv')

# Pre-elaborazione dei dati: rimuove righe con valori mancanti
adult_data = adult_data.replace(' ?', pd.NA).dropna()

# Codifica delle variabili categoriche
adult_data = pd.get_dummies(adult_data, drop_first=True)

# Definizione della variabile target e delle feature sensibili
target = 'income_>50K'
sensitive_features = ['sex_Male']
y = adult_data[target]
X = adult_data.drop(columns=[target] + sensitive_features)
X = StandardScaler().fit_transform(X)  # Standardizzazione delle feature

# Divisione del dataset
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, adult_data[sensitive_features], test_size=0.2, random_state=42
)

# Feature sensibile per il test set
sensitive_feature = sensitive_test['sex_Male'].values

# Definizione dei modelli
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "XGBoost": XGBClassifier()
}

# Addestramento e valutazione dei modelli
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_test = y_test.reset_index(drop=True)
    predictions = pd.Series(predictions).reset_index(drop=True)
    # Inizializzazione del toolkit con predizioni, etichette e feature sensibile
    toolkit = Toolkit(predictions, y_test.reset_index(drop=True), sensitive_feature)
    
    print(model_name)
    print("modello senza privacy:")
    toolkit.summary_fairness_accuracy()

    # Valutazione del trade-off tra accuratezza e fairness
    toolkit.evaluate_tradeoff_accuracy_fairness(noise_type='laplace', data_type='categorical', delta=0.1)


