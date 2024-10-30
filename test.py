from privacy_fairness_toolkit import Toolkit

data = {
        'predictions': [2,0,1,2],
        'actuals': [2,1,1,2],
        'gender': ['F','M','F','F'],
        'race': ['A', 'B','A']
    }

t = Toolkit(data['actuals'],data['predictions'],data['gender'])
Toolkit.evaluate_tradeoff_accuracy_fairness(t,[0.1,2,5,10])



