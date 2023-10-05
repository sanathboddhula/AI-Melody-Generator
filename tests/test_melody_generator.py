import unittest
import torch
from melody_generator import MelodyGenerator, train_model

"""
Unit Tests for Melody Generation Code

This file contains a suite of unit tests for the Melody Generation code implemented using PyTorch. The tests cover critical components such as model initialization, forward pass, training, and model saving/loading.

Usage:
1. Ensure that the `melody_generator.py` file is available in the same directory.
2. Run the unit tests using the following command:
    python -m unittest tests.test_melody_generator 
"""


class TestMelodyGenerator(unittest.TestCase):
    """
    Unit Tests for MelodyGenerator Class and Training Process

    This test suite covers the functionality of the MelodyGenerator class and its training process.

    Test Methods:
        - test_model_initialization: Verify the model initializes correctly.
        - test_forward_pass: Test the forward pass of the model.
        - test_training: Verify the training process.
        - test_save_and_load_model: Test model saving and loading.
    """

    def test_model_initialization(self):
        """
        Test Model Initialization

        This test checks if the MelodyGenerator class initializes without errors and creates an instance of the model.
        """
        model = MelodyGenerator(vocab_size=128, embedding_dim=64, hidden_units=128)
        self.assertIsInstance(model, MelodyGenerator)

    def test_forward_pass(self):
        """
        Test Forward Pass

        This test verifies that the model's forward pass produces the expected output shape.
        """
        model = MelodyGenerator(vocab_size=128, embedding_dim=64, hidden_units=128)
        input_data = torch.LongTensor([[1, 2, 3, 4, 5]])
        output = model(input_data)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 5, 128))  # Adjust the shape as per your model's output.

    def test_training(self):
        """
        Test Model Training

        This test checks if the training process executes without errors. It can be extended to include loss assertions.
        """
        model = MelodyGenerator(vocab_size=128, embedding_dim=64, hidden_units=128)
        input_sequences = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        target_sequences = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        train_model(model, input_sequences, target_sequences, batch_size=2, epochs=1)

        # You can add assertions to check if training succeeded (e.g., loss decreases).

    def test_save_and_load_model(self):
        """
        Test Model Saving and Loading

        This test ensures that the model can be successfully saved to and loaded from a file.
        """
        model = MelodyGenerator(vocab_size=128, embedding_dim=64, hidden_units=128)
        torch.save(model.state_dict(), 'test_model.pt')
        loaded_model = MelodyGenerator(vocab_size=128, embedding_dim=64, hidden_units=128)
        loaded_model.load_state_dict(torch.load('test_model.pt'))
        self.assertIsInstance(loaded_model, MelodyGenerator)


if __name__ == '__main__':
    unittest.main()
