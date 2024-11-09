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

    # Metodo per applicare la differential privacy a una variabile categorica tramite Laplace
    def apply_pure_categorical_dp(self, epsilon, delta=0.1):
        """Applica privacy differenziale al modello, aggiungendo rumore alle predizioni."""
        dp = DifferentialPrivacy(epsilon, delta)
        return dp.add_laplace_categorical_noise(self.predictions)
    # Metodo per applicare la differential privacy a una variabile quantitativa tramite Laplace
    def apply_pure_dp(self, epsilon, delta=0.1):
        """Applica privacy differenziale al modello, aggiungendo rumore alle predizioni."""
        dp = DifferentialPrivacy(epsilon, delta)
        return dp.add_laplace_noise(self.predictions)
    # Metodo per applicare la delta differential privacy a una variabile categorica tramite Gauss
    def apply_categorical_delta_dp(self, epsilon, delta=0.1):
        """Applica privacy differenziale al modello, aggiungendo rumore alle predizioni."""
        dp = DifferentialPrivacy(epsilon, delta)
        return dp.add_gaussian_categorical_noise(self.predictions)
     # Metodo per applicare la delta differential privacy a una variabile quantitativa tramite Gauss
    def apply_delta_dp(self, epsilon, delta=0.1):
        """Applica privacy differenziale al modello, aggiungendo rumore alle predizioni."""
        dp = DifferentialPrivacy(epsilon, delta)
        return dp.add_gaussian_noise(self.predictions)
    
    
        
    
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
        


    def summary_fairness_accuracy(self):
        em = ModelEvaluator(self.labels,self.predictions)
        fm = FairnessMetrics(self.predictions, self.labels,self.sensitive_features)
        print("accuracy: " + str(em.accuracy()))
        print("demographic_parity: " + str(fm.compute_statistical_parity()))
        print("predictive_parity: " + str(fm.compute_predictive_parity()))
        print("compute_equal_opportunity: " + str(fm.compute_equal_opportunity()))
        print("compute_well_calibration: " + str(fm.compute_well_calibration()))
        
        
    def evaluate_tradeoff_accuracy_fairness(self, noise_type, data_type, delta=0.1):
     """
     Calcola l'accuratezza e le metriche di fairness per una gamma di valori di epsilon,
     in modo da valutare il trade-off tra privacy, accuratezza e fairness.

     :param epsilon_values: lista di valori epsilon per la privacy differenziale
     :param noise_type: tipo di rumore ('laplace' o 'gaussian')
     :param data_type: tipo di dato ('categorical' o 'quantitative')
     :param delta: valore delta per la privacy differenziale
     :return: dizionario contenente accuracy e metriche di fairness per ciascun valore di epsilon
     """
     tradeoff_results = {}
     epsilon_values=[0.1,0.3,0.7,1.5,2,2.5,3,5,7]
     
     for epsilon in epsilon_values:
        # Scegliere il metodo appropriato in base a `noise_type` e `data_type`
        if noise_type == 'laplace' and data_type == 'categorical':
            noisy_predictions = self.apply_pure_categorical_dp(epsilon, delta)
        elif noise_type == 'laplace' and data_type == 'quantitative':
            noisy_predictions = self.apply_pure_dp(epsilon, delta)
        elif noise_type == 'gaussian' and data_type == 'categorical':
            noisy_predictions = self.apply_categorical_delta_dp(epsilon, delta)
        elif noise_type == 'gaussian' and data_type == 'quantitative':
            noisy_predictions = self.apply_delta_dp(epsilon, delta)
        else:
            raise ValueError("Invalid combination of noise_type and data_type")
        
        em = ModelEvaluator(self.labels,noisy_predictions)

        # Calcola le metriche di fairness con FairnessMetrics
        fairness_evaluator = FairnessMetrics(noisy_predictions, self.labels, self.sensitive_features)
        demographic_parity = fairness_evaluator.compute_statistical_parity()
        equal_opportunity= fairness_evaluator.compute_equal_opportunity()
        well_calibration= fairness_evaluator.compute_well_calibration()
        accuracy =  em.accuracy()
        predictive_parity = fairness_evaluator.compute_predictive_parity()

        # Salva i risultati nel dizionario per questo valore di epsilon
        tradeoff_results[epsilon] = {
            "accuracy":accuracy,
            "demographic_parity": demographic_parity,
            "equal_opportunity": equal_opportunity,
            "predictive_parity": predictive_parity,
            "well_calibration": well_calibration
        }

     Utilities.print_dictionary(tradeoff_results)
     return tradeoff_results
    
