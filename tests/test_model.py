import unittest
from src.model_training import train_model

class TestModelTraining(unittest.TestCase):
    
    def test_train_model(self):
        model, metrics = train_model()
        self.assertIsNotNone(model)
        self.assertIsInstance(metrics, dict)
        
if __name__ == '__main__':
    unittest.main()
