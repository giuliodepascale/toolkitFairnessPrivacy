class FairnessMetrics:
 def __init__(self, predictions, labels,sensitive_features):
        self.predictions = predictions
        self.labels = labels
        self.sensitive_features = sensitive_features

 def compute_statistical_parity(self):
        """
        Calcola la parità demografica per gruppi protetti generici.

        :return: La differenza massima di parità statistica tra i gruppi.
        """
        unique_groups = set(self.sensitive_features)
        group_parity = {}

        for group in unique_groups:
            # Filtra le predizioni per il gruppo attuale
            group_predictions = [
                self.predictions[i]
                for i in range(len(self.sensitive_features))
                if self.sensitive_features[i] == group
            ]
            # Calcola la proporzione di predizioni positive per il gruppo
            group_parity[group] = (
                sum(group_predictions) / len(group_predictions) if group_predictions else 0
            )

        # Calcola la differenza massima tra le parità dei gruppi
        max_parity_difference = max(group_parity.values()) - min(group_parity.values())
        return max_parity_difference
    
 def compute_equalized_odds(self):
    """Calcola la metrica Equalized Odds per più classi."""
    unique_groups = set(self.sensitive_features)
    metrics = {group: {"tpr": 0, "fpr": 0, "true_positive": 0, "false_positive": 0} for group in unique_groups}

    for i in range(len(self.sensitive_features)):
        pred, label, group = self.predictions[i], self.labels[i], self.sensitive_features[i]
        
        if label == 1 and pred == 1:
            metrics[group]["true_positive"] += 1  # True Positive
        elif label == 0 and pred == 1:
            metrics[group]["false_positive"] += 1  # False Positive

    # Calculate TPR and FPR for each group
    for group in unique_groups:
        total_positives = sum(1 for i in range(len(self.labels)) if self.labels[i] == 1 and self.sensitive_features[i] == group)
        total_negatives = sum(1 for i in range(len(self.labels)) if self.labels[i] == 0 and self.sensitive_features[i] == group)

        metrics[group]["tpr"] = metrics[group]["true_positive"] / total_positives if total_positives else 0
        metrics[group]["fpr"] = metrics[group]["false_positive"] / total_negatives if total_negatives else 0

    # Calculate differences
    # Calculate differences
    differences = {}
    # Convert the set to a list to access elements by index
    unique_groups_list = list(unique_groups)

    for i, group in enumerate(unique_groups_list):
        for other_group in unique_groups_list[i + 1:]:  # Only consider groups after the current one
            differences[(group, other_group)] = {
            "tpr_difference": abs(metrics[group]["tpr"] - metrics[other_group]["tpr"]),
            "fpr_difference": abs(metrics[group]["fpr"] - metrics[other_group]["fpr"]),
        }

    return differences






 def compute_predictive_parity(self):
    """
    Calcola la metrica Predictive Parity.
    
    :return: Differenza nella precisione tra i gruppi
    """
    # Identifica i gruppi unici basati sulle caratteristiche sensibili
    unique_groups = set(self.sensitive_features)

    # Crea un dizionario per memorizzare le precisioni per ogni gruppo
    group_precisions = {}

    for group in unique_groups:
        group_correct = sum(1 for i in range(len(self.predictions))
                            if self.predictions[i] == self.labels[i] and self.sensitive_features[i] == group)
        group_total = sum(1 for i in range(len(self.sensitive_features)) 
                          if self.sensitive_features[i] == group)

        # Calcola la precisione per il gruppo
        precision_group = group_correct / group_total if group_total > 0 else 0
        group_precisions[group] = precision_group

    # Calcola la differenza di precisione tra i gruppi
    precision_values = list(group_precisions.values())
    if len(precision_values) > 1:
        return abs(precision_values[0] - precision_values[1])  # Return difference between first two groups
    else:
        return 0  # Not enough groups to calculate difference
 
 def compute_accuracy_parity(self):
        """Calculate accuracy parity for protected groups."""
        unique_groups = set(self.sensitive_features)
        group_accuracy = {}
        
        for group in unique_groups:
            group_correct = sum(1 for i in range(len(self.predictions)) 
                                if self.predictions[i] == self.labels[i] and self.sensitive_features[i] == group)
            group_total = sum(1 for i in range(len(self.sensitive_features)) 
                              if self.sensitive_features[i] == group)
            group_accuracy[group] = group_correct / group_total if group_total > 0 else 0
        
        return max(group_accuracy.values()) - min(group_accuracy.values())

 def compute_false_positive_parity(self):
        """Calculate false positive parity for protected groups."""
        unique_groups = set(self.sensitive_features)
        group_false_positive_rate = {}

        for group in unique_groups:
            false_positives = sum(1 for i in range(len(self.predictions))
                                  if self.predictions[i] == 1 and self.labels[i] == 0 and self.sensitive_features[i] == group)
            total_negatives = sum(1 for i in range(len(self.sensitive_features))
                                  if self.sensitive_features[i] == group and self.labels[i] == 0)
            group_false_positive_rate[group] = false_positives / total_negatives if total_negatives > 0 else 0

        return max(group_false_positive_rate.values()) - min(group_false_positive_rate.values())
  
 def compute_positive_rate_parity(self):
        """Calculate positive rate parity for protected groups."""
        unique_groups = set(self.sensitive_features)
        group_positive_rate = {}

        for group in unique_groups:
            positives = sum(1 for i in range(len(self.predictions))
                            if self.predictions[i] == 1 and self.sensitive_features[i] == group)
            total_for_group = sum(1 for i in range(len(self.sensitive_features))
                                  if self.sensitive_features[i] == group)
            group_positive_rate[group] = positives / total_for_group if total_for_group > 0 else 0

        return max(group_positive_rate.values()) - min(group_positive_rate.values())

 def compute_predictive_value_parity(self):
        """Calculate predictive value parity for protected groups."""
        unique_groups = set(self.sensitive_features)
        group_predictive_value = {}

        for group in unique_groups:
            true_positives = sum(1 for i in range(len(self.predictions))
                                 if self.predictions[i] == 1 and self.labels[i] == 1 and self.sensitive_features[i] == group)
            total_predicted_positive = sum(1 for i in range(len(self.sensitive_features))
                                            if self.sensitive_features[i] == group and self.predictions[i] == 1)
            true_negatives = sum(1 for i in range(len(self.predictions))
                                 if self.predictions[i] == 0 and self.labels[i] == 0 and self.sensitive_features[i] == group)
            total_predicted_negative = sum(1 for i in range(len(self.sensitive_features))
                                            if self.sensitive_features[i] == group and self.predictions[i] == 0)

            ppv = true_positives / (true_positives + total_predicted_positive) if total_predicted_positive > 0 else 0
            npv = true_negatives / (true_negatives + total_predicted_negative) if total_predicted_negative > 0 else 0
            
            group_predictive_value[group] = (ppv, npv)

        ppv_values = [values[0] for values in group_predictive_value.values()]
        npv_values = [values[1] for values in group_predictive_value.values()]

        return (max(ppv_values) - min(ppv_values), max(npv_values) - min(npv_values))

 def compute_equal_opportunity(self):
        """Calculate equal opportunity for protected groups."""
        unique_groups = set(self.sensitive_features)
        group_equal_opportunity = {}

        for group in unique_groups:
            true_positives = sum(1 for i in range(len(self.predictions))
                                 if self.predictions[i] == 1 and self.labels[i] == 1 and self.sensitive_features[i] == group)
            total_actual_positives = sum(1 for i in range(len(self.sensitive_features))
                                          if self.sensitive_features[i] == group and self.labels[i] == 1)
            group_equal_opportunity[group] = true_positives / total_actual_positives if total_actual_positives > 0 else 0

        return max(group_equal_opportunity.values()) - min(group_equal_opportunity.values())

 def compute_well_calibration(self):
        """Calculate well calibration for protected groups."""
        unique_groups = set(self.sensitive_features)
        group_calibration = {}

        for group in unique_groups:
            predicted_probabilities = [self.predictions[i] for i in range(len(self.sensitive_features)) if self.sensitive_features[i] == group]
            actuals = [self.labels[i] for i in range(len(self.sensitive_features)) if self.sensitive_features[i] == group]

            # Simple calibration check
            group_calibration[group] = abs(sum(predicted_probabilities) / len(predicted_probabilities) - sum(actuals) / len(actuals)) if len(predicted_probabilities) > 0 else 0

        return max(group_calibration.values()) - min(group_calibration.values())

 def compute_balance_for_positive_class(self):
        """Calculate balance for positive class for protected groups."""
        unique_groups = set(self.sensitive_features)
        group_balance = {}

        for group in unique_groups:
            expected_positive = sum(1 for i in range(len(self.sensitive_features)) 
                                    if self.sensitive_features[i] == group and self.labels[i] == 1)
            total_positive = sum(1 for i in range(len(self.sensitive_features)) 
                                 if self.labels[i] == 1)
            group_balance[group] = expected_positive / total_positive if total_positive > 0 else 0

        return max(group_balance.values()) - min(group_balance.values())

 def compute_balance_for_negative_class(self):
        """Calculate balance for negative class for protected groups."""
        unique_groups = set(self.sensitive_features)
        group_balance = {}

        for group in unique_groups:
            expected_negative = sum(1 for i in range(len(self.sensitive_features)) 
                                    if self.sensitive_features[i] == group and self.labels[i] == 0)
            total_negative = sum(1 for i in range(len(self.sensitive_features)) 
                                 if self.labels[i] == 0)
            group_balance[group] = expected_negative / total_negative if total_negative > 0 else 0

        return max(group_balance.values()) - min(group_balance.values())

