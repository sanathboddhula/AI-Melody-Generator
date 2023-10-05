import unittest
import os
import mido
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from notebooks.data_preprocessing import MIDIDataProcessor


class TestMIDIDataProcessor(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'D:/IdeaProjects/midi-storage'
        self.processor = MIDIDataProcessor(self.data_dir)

    def test_quantize_timing(self):
        """
        Tests the quantize_timing method.
        """
        events = [(0.0, 60, 0.5), (0.7, 62, 0.4), (1.2, 64, 0.3)]
        quantized_events = self.processor.quantize_timing(events)
        self.assertEqual(quantized_events, [(0.0, 60, 0.5), (0.75, 62, 0.4), (1.25, 64, 0.3)])

    def test_preprocess_data(self):
        """
        Tests the preprocess_data method.
        """
        sequences, tempos = self.processor.preprocess_data()
        self.assertIsInstance(sequences, list)
        self.assertIsInstance(tempos, list)

    def test_create_input_sequences(self):
        """
        Tests the create_input_sequences method.
        """
        sequences, _ = self.processor.preprocess_data()
        input_sequences = self.processor.create_input_sequences(sequence_length=4)
        self.assertIsInstance(input_sequences, list)
        self.assertGreater(len(input_sequences), 0)

    def test_normalize_data(self):
        """
        Tests the normalize_data method.
        """
        sequences, _ = self.processor.preprocess_data()
        input_sequences = self.processor.create_input_sequences(sequence_length=4)
        scaler = self.processor.normalize_data(input_sequences)
        self.assertIsInstance(scaler, StandardScaler)

    def test_split_data(self):
        """
        Tests the split_data method.
        """
        sequences, tempos = self.processor.preprocess_data()
        input_sequences = self.processor.create_input_sequences(sequence_length=4)
        X_train, X_test, y_train, y_test = self.processor.split_data(input_sequences, tempos)
        self.assertIsInstance(X_train, list)
        self.assertIsInstance(X_test, list)
        self.assertIsInstance(y_train, list)
        self.assertIsInstance(y_test, list)


if __name__ == '__main__':
    unittest.main()
