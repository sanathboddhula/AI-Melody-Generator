import argparse
import os
import torch
import numpy as np
from torch import nn

# Define the MelodyGenerator class for melody generation
class MelodyGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(MelodyGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_units, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_units, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

    def generate_melody(self, genre, mood, tempo, scale):
        """
        Generate Melody Based on User Preferences

        Args:
            genre (str): The genre of the melody.
            mood (str): The mood or emotion of the melody.
            tempo (int): The tempo of the melody in BPM.
            scale (str): The musical scale for the melody.

        Returns:
            str: The generated melody as a string (MIDI format).
        """
        # Your melody generation logic based on user preferences here
        # This method should return the generated melody as a string in MIDI format

        # Example: Generate a simple MIDI melody
        generated_melody = "MIDI melody generated based on user preferences."

        return generated_melody

    def save_midi(self, melody, output_directory, filename="generated_melody.mid"):
        """
        Save Melody as a MIDI File

        Args:
            melody (str): The melody in MIDI format as a string.
            output_directory (str): The directory where the MIDI file will be saved.
            filename (str, optional): The name of the output MIDI file (default is "generated_melody.mid").
        """
        # Ensure the output directory exists; create it if it doesn't
        os.makedirs(output_directory, exist_ok=True)

        # Construct the full path to the output MIDI file
        output_filepath = os.path.join(output_directory, filename)

        # Save the generated melody as a MIDI file
        with open(output_filepath, "w") as midi_file:
            midi_file.write(melody)

        print(f"Generated melody saved as {output_filepath}")

# Define the CLI for user input
def parse_command_line_arguments():
    """
    Parse Command-Line Arguments

    This function defines and parses command-line arguments for the AI Melody Generator CLI.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="AI Melody Generator CLI")

    parser.add_argument("--output_directory", type=str, default="D:/IdeaProjects/AI Melody Generator/data/generated_melodies",
                        help="Specify the directory where MIDI files will be saved")

    return parser.parse_args()

def main():
    """
    Main Function for the AI Melody Generator CLI

    This function is the entry point of the CLI. It parses command-line arguments, interacts with the MelodyGenerator
    class, generates melodies based on user preferences, and saves them to the specified directory.
    """
    # Parse command-line arguments
    args = parse_command_line_arguments()

    # Create an instance of the MelodyGenerator
    melody_generator = MelodyGenerator(vocab_size=128, embedding_dim=64, hidden_units=128)

    # Define a list of user preferences (you can customize this)
    user_preferences = [
        {"genre": "Pop", "mood": "Happy", "tempo": 120, "scale": "C Major"},
        {"genre": "Rock", "mood": "Energetic", "tempo": 140, "scale": "A Minor"},
        # Add more user preferences as needed
    ]

    # Generate melodies based on different user preferences and save them to the specified directory
    for idx, preference in enumerate(user_preferences, start=1):
        generated_melody = melody_generator.generate_melody(**preference)
        filename = f"generated_melody_{idx}.mid"
        melody_generator.save_midi(generated_melody, args.output_directory, filename)

if __name__ == '__main__':
    main()
