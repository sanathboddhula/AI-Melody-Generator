"""
Melody Generation using PyTorch

This script trains a deep learning model to generate melodies using PyTorch. The model is designed to predict the next note in a sequence of MIDI notes. The training data consists of input sequences and target sequences, where the target is the next note after the input sequence.

Requirements:
- PyTorch (https://pytorch.org/)
- NumPy

Usage:
1. Define or replace the `midi_dataset` with your MIDI dataset.
2. Adjust hyperparameters such as sequence length, embedding dimension, hidden units, batch size, and epochs.
3. Run the script to train the model.
4. The trained model will be saved as 'melody_generator_model.pt' and can be used for melody generation.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MelodyGenerator(nn.Module):
    """
    Melody Generator Model

    This class defines the melody generation model using PyTorch.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique MIDI note values).
        embedding_dim (int): Dimension of the embedding layer.
        hidden_units (int): Number of hidden units in the LSTM layer.

    Methods:
        forward(x): Forward pass of the model.

    Example:
        model = MelodyGenerator(vocab_size, embedding_dim, hidden_units)
        output = model(input_data)
    """

    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(MelodyGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, vocab_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, vocab_size).
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output


def train_model(model, input_sequences, target_sequences, batch_size, epochs):
    """
    Train the melody generation model.

    Args:
        model (MelodyGenerator): Melody generation model to be trained.
        input_sequences (list): List of input sequences (each sequence is a list of MIDI note values).
        target_sequences (list): List of target sequences (each sequence is a single MIDI note value).
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        for i in range(0, len(input_sequences), batch_size):
            inputs = torch.LongTensor(input_sequences[i:i + batch_size])
            targets = torch.LongTensor(target_sequences[i:i + batch_size])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(2)), targets.view(-1))
            loss.backward()
            optimizer.step()

            if (i // batch_size) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i // batch_size + 1}], Loss: {loss.item():.4f}')

    print('Training finished.')


# Example usage
if __name__ == "__main__":
    # Sample MIDI dataset (replace with your dataset)
    midi_dataset = [
        [60, 62, 64, 65, 67, 69],
        [62, 64, 65, 67, 69, 71],
        # Add more sequences here
    ]

    sequence_length = 6
    vocab_size = 128
    embedding_dim = 64
    hidden_units = 128
    batch_size = 32
    epochs = 50

    input_sequences = []
    target_sequences = []
    for sequence in midi_dataset:
        for i in range(len(sequence) - sequence_length):
            input_seq = sequence[i:i + sequence_length]
            target_seq = sequence[i + sequence_length]
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

    melody_generator = MelodyGenerator(vocab_size, embedding_dim, hidden_units)
    train_model(melody_generator, input_sequences, target_sequences, batch_size, epochs)

    # Save the trained model to a file (PyTorch's .pt format)
    torch.save(melody_generator.state_dict(), 'melody_generator_model.pt')