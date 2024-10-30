import random
import math
import numpy as np
from utilities import Utilities

class DifferentialPrivacy:
    def __init__(self, epsilon, delta=0.1):
        self.epsilon = epsilon
        self.delta = delta

    def add_laplace_noise(self, values, sensitivity=1):
        """
        Aggiunge rumore di Laplace ai valori per garantire l'epsilon-Differential Privacy.

        :param values: Lista di valori ai quali si vuole aggiungere il rumore
        :param sensitivity: Sensitività della funzione, ovvero il massimo cambiamento nell'output
                            dovuto alla modifica di un singolo record (default = 1)
        :return: Lista di valori con rumore laplaciano aggiunto
        """
        # Calcola il parametro di scala in base a epsilon
        scale = sensitivity / self.epsilon
        # Genera rumore di Laplace con media 0 e parametro di scala calcolato
        noisy_values = [v + random.gauss(0, scale) for v in values]
        return noisy_values
    
    def add_gaussian_noise(self, values, sensitivity=1):
        """
        Aggiunge rumore gaussiano ai valori per garantire l'(epsilon, delta)-Differential Privacy.

        :param values: Lista di valori ai quali si vuole aggiungere il rumore
        :param sensitivity: Sensitività della funzione, ovvero il massimo cambiamento nell'output
                            dovuto alla modifica di un singolo record (default = 1)
        :return: Lista di valori con rumore gaussiano aggiunto
        """
        # Calcola sigma in base a epsilon e delta
        sigma = math.sqrt(2 * math.log(1.25 / self.delta)) * sensitivity / self.epsilon
        # Genera rumore gaussiano con media 0 e deviazione standard sigma
        noisy_values = [v + random.gauss(0, sigma) for v in values]
        return noisy_values
    
    def add_laplace_categorical_noise(self, values):
     """
     Simula rumore Laplaciano per variabili categoriche senza richiedere la lista delle categorie.
 
     :param values: Lista di valori categorici
     :return: Lista con rumore simulato per le variabili categoriche
     """
     # Ottieni i valori unici dai dati
     categories = list(set(values))

     noisy_values = []
     for value in values:
         probabilities = []
         for cat in categories:
             # Probabilità decresce esponenzialmente con la "distanza" simulata
             distance = 1 if cat != value else 0
             probabilities.append(math.exp(-distance * self.epsilon))
         # Normalizza per ottenere una distribuzione di probabilità
         probabilities = np.array(probabilities) / np.sum(probabilities)
         # Campionamento in base alla distribuzione
         noisy_value = np.random.choice(categories, p=probabilities)
         noisy_values.append(noisy_value)
         noisy_values = [Utilities.convert_to_native(noisy_value) for noisy_value in noisy_values]
     return noisy_values

    def add_gaussian_categorical_noise(self, values, sensitivity=1.0):
     """
     Simula rumore Gaussiano per variabili categoriche senza richiedere la lista delle categorie.
    
     :param values: Lista di valori categorici
     :param sensitivity: Sensitività della funzione, ovvero il massimo cambiamento nell'output dovuto 
                        alla modifica di un singolo record (default = 1.0)
     :return: Lista con rumore simulato per le variabili categoriche
     """
     # Ottieni i valori unici dai dati
     categories = list(set(values))

     noisy_values = []
     for value in values:
        probabilities = []
        for cat in categories:
            # Calcola la distanza tra la categoria corrente e quella osservata
            distance = 1 if cat != value else 0

            # Probabilità decresce con un profilo gaussiano (centrato sulla categoria originale)
            # Dove il denominatore è legato alla varianza che aumenta al diminuire di epsilon
            variance = (2 * (sensitivity ** 2)) / (self.epsilon ** 2)
            probabilities.append(math.exp(- (distance ** 2) / (2 * variance)))

        # Normalizza per ottenere una distribuzione di probabilità
        probabilities = np.array(probabilities) / np.sum(probabilities)

         # Campionamento in base alla distribuzione
        noisy_value = np.random.choice(categories, p=probabilities)

        # Aggiungi il valore disturbato alla lista
        noisy_values.append(noisy_value)

     # Converti tutti i valori noisy in tipi nativi Python, se necessario
     noisy_values = [Utilities.convert_to_native(noisy_value) for noisy_value in noisy_values]

     return noisy_values
   