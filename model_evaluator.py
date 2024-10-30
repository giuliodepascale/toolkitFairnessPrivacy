class ModelEvaluator:
    def __init__(self, y_true, y_pred):
        """
        Inizializza la classe con i valori veri e i valori predetti.

        :param y_true: valori veri
        :param y_pred: valori predetti
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.tn = 0  # True Negatives
        self._calculate_confusion_matrix()

    def _calculate_confusion_matrix(self):
        """Calcola manualmente i valori della matrice di confusione (tp, fp, fn, tn)."""
        for true, pred in zip(self.y_true, self.y_pred):
            if true == 1 and pred == 1:
                self.tp += 1
            elif true == 0 and pred == 1:
                self.fp += 1
            elif true == 1 and pred == 0:
                self.fn += 1
            elif true == 0 and pred == 0:
                self.tn += 1

    def accuracy(self):
        """Calcola l'accuratezza del modello."""
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 0

    def precision(self):
        """Calcola la precisione del modello."""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0

    def recall(self):
        """Calcola il richiamo del modello."""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0

    def f1(self):
        """Calcola l'F1 score del modello."""
        precision = self.precision()
        recall = self.recall()
        return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    def confusion_matrix(self):
        """Restituisce la matrice di confusione come una lista di liste."""
        return [[self.tn, self.fp], [self.fn, self.tp]]

    def summary(self):
        """Ritorna un dizionario con tutte le metriche calcolate."""
        return {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(),
            'Recall': self.recall(),
            'F1 Score': self.f1(),
            'Confusion Matrix': self.confusion_matrix()
        }
