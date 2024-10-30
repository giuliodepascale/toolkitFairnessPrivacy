from differential_privacy import DifferentialPrivacy
from fairness_metrics import FairnessMetrics
from model_evaluator import ModelEvaluator
from utilities import Utilities


class Toolkit:
    def __init__(self, predictions, labels, sensitive_features):
        """
        Inizializza il toolkit con un modello di ML e dati necessari per calcolare privacy e fairness.
        
        :param predictions: Predizioni del modello
        :param labels: Etichette reali
        :param sensitive_features: Feature sensibili per la fairness (es. genere, etnia)
        """
        self.predictions = predictions
        self.labels = labels
        self.sensitive_features = sensitive_features

    # Metodo per calcolare le metriche di privacy
    def apply_differential_privacy(self, epsilon, delta=0.1):
        """Applica privacy differenziale al modello, aggiungendo rumore alle predizioni."""
        dp = DifferentialPrivacy(epsilon, delta)
        return dp.add_laplace_categorical_noise(self.predictions)
        
    
     # Metodo per calcolare le metriche di fairness
    def summary_fairness_metrics(self):
        fm = FairnessMetrics(self.predictions, self.labels,self.sensitive_features)
        print("demographic_parity: " + str(fm.compute_statistical_parity()))
        print("equalized_odds: " + str(fm.compute_equalized_odds()))
        print("predictive_parity: " + str(fm.compute_predictive_parity()))
        print("compute_predictive_value_parity: " + str(fm.compute_predictive_value_parity()))
        print("compute_positive_rate_parity: " + str(fm.compute_positive_rate_parity()))
        print("compute_false_positive_parity: " + str(fm.compute_false_positive_parity()))
        print("compute_equal_opportunity: " + str(fm.compute_equal_opportunity()))
        print("compute_well_calibration: " + str(fm.compute_well_calibration()))
        print("compute_balance_for_positive_class: " + str(fm.compute_balance_for_positive_class()))
        print("compute_balance_for_negative_class: " + str(fm.compute_balance_for_negative_class()))
        
        
    def summary_evaluation_metrics(self):
        em = ModelEvaluator(self.labels,self.predictions)
        print(em.summary())
        
    def evaluate_tradeoff_accuracy_fairness(self, epsilon_values, delta=0.1):
     """
     Calcola l'accuratezza e le metriche di fairness per una gamma di valori di epsilon,
     in modo da valutare il trade-off tra privacy, accuratezza e fairness.

     :param epsilon_values: lista di valori epsilon per la privacy differenziale
     :param delta: valore delta per la privacy differenziale
     :return: dizionario contenente accuracy e metriche di fairness per ciascun valore di epsilon
     """
     tradeoff_results = {}

     for epsilon in epsilon_values:
        # Applica la privacy differenziale alle predizioni
        noisy_predictions = self.apply_differential_privacy(epsilon, delta)

        # Calcola l'accuratezza con ModelEvaluator
        evaluator = ModelEvaluator(self.labels, noisy_predictions)
        accuracy = evaluator.accuracy()

        # Calcola le metriche di fairness con FairnessMetrics
        fairness_evaluator = FairnessMetrics(noisy_predictions, self.labels, self.sensitive_features)
        demographic_parity = fairness_evaluator.compute_statistical_parity()
        equalized_odds = fairness_evaluator.compute_equalized_odds()
        predictive_parity = fairness_evaluator.compute_predictive_parity()

        # Salva i risultati nel dizionario per questo valore di epsilon
        tradeoff_results[epsilon] = {
            "accuracy": accuracy,
            "demographic_parity": demographic_parity,
            "equalized_odds": equalized_odds,
            "predictive_parity": predictive_parity
        }

     Utilities.print_dictionary(tradeoff_results)
     return tradeoff_results
