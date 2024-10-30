import numpy as np
class Utilities:
    

  def convert_to_native(value):
    """
    Converte un valore numpy o altri tipi in un tipo di dato nativo Python.

    Questo metodo verifica se il valore fornito è un tipo numpy scalare (es. np.int64, np.float64, np.bool_, np.str_)
    e lo converte nel corrispondente tipo di dato nativo Python (es. int, float, bool, str).
    In alternativa, se il valore è un array numpy, lo converte in una lista Python.
    Se il valore è già un tipo nativo Python, lo restituisce senza modifiche.

    Parameters:
    value (Any): Il valore che si desidera convertire. Può essere un valore scalare numpy, un array numpy,
                 oppure un tipo nativo Python (int, float, str, bool).

    Returns:
    Any: Il valore convertito in un tipo di dato nativo Python, se possibile.
         Ad esempio, numpy.int64 diventa int, numpy.ndarray diventa list, ecc.
    """
    # Verifica se il valore è un oggetto numpy scalare (come np.int64, np.float64, np.bool_, np.str_)
    if isinstance(value, (np.integer, np.floating, np.bool_, np.str_)):
        return value.item()
    # Se è già un tipo Python nativo (int, float, str, bool), ritorna semplicemente il valore
    elif isinstance(value, (int, float, str, bool)):
        return value
    # Se il valore è un array numpy (non scalare), convertilo in una lista Python
    elif isinstance(value, np.ndarray):
        return value.tolist()
    # In tutti gli altri casi, ritorna il valore stesso senza modifica
    else:
        return value
    
  def print_dictionary(d):
    """
    Stampa un dizionario in modo leggibile, andando a capo per ogni campo.

    :param d: dizionario da stampare
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")