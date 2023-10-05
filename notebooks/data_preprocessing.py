import os
import mido
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MIDIDataProcessor:
    """
    A class for preprocessing MIDI data, including extracting note values, durations, tempo, and more.

    Args:
        midi_data_dir (str): The directory path where MIDI files are located.
        time_grid (float, optional): The time grid for quantizing timing (default is 0.25, representing quarter notes).

    Attributes:
        note_to_int (dict): A dictionary mapping musical notes to unique integer values.
        all_sequences (list): A list of preprocessed MIDI sequences, each representing a melody.
        all_tempos (list): A list of tempo values extracted from MIDI files.

    Methods:
        quantize_timing(events): Quantizes the timing of MIDI events to the specified time grid.
        preprocess_data(): Loads and preprocesses MIDI data from the specified directory.
        create_input_sequences(sequence_length): Creates input sequences with a fixed length.
        normalize_data(input_sequences): Normalizes data (note values and durations) using standard scaling.
        split_data(input_sequences, tempos, test_size=0.2, random_state=42): Splits data into training and testing sets.

    Usage:
        # Initialize the data processor
        data_processor = MIDIDataProcessor('path/to/midi/data')

        # Preprocess the MIDI data
        all_sequences, all_tempos = data_processor.preprocess_data()

        # Create input sequences for model training
        input_sequences = data_processor.create_input_sequences(sequence_length)

        # Normalize data (optional)
        scaler = data_processor.normalize_data(input_sequences)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = data_processor.split_data(input_sequences, all_tempos)
    """

    def __init__(self, midi_data_dir, time_grid=0.25):
        self.midi_data_dir = midi_data_dir
        self.time_grid = time_grid
        self.note_to_int = {}
        self.all_sequences = []
        self.all_tempos = []

    def quantize_timing(self, events):
        quantized_events = []
        current_time = 0.0

        for event in events:
            time, note_value, duration = event
            delta_time = time - current_time
            quantized_delta_time = round(delta_time / self.time_grid) * self.time_grid
            quantized_time = current_time + quantized_delta_time
            quantized_events.append([quantized_time, note_value, duration])
            current_time = quantized_time

        return quantized_events

    def preprocess_data(self):
        for midi_file in os.listdir(self.midi_data_dir):
            if midi_file.endswith('.mid'):
                midi_path = os.path.join(self.midi_data_dir, midi_file)

                # Load the MIDI file
                midi = mido.MidiFile(midi_path)

                # Initialize variables to store note and duration data for the current MIDI file
                notes = []
                durations = []

                # Extract tempo information (assuming one tempo change event)
                tempo = None
                for track in midi.tracks:
                    for msg in track:
                        if msg.type == 'set_tempo':
                            tempo = mido.tempo2bpm(msg.tempo)
                            break
                    if tempo:
                        break

                # Extract note and duration events from the MIDI file
                current_time = 0.0
                for track in midi.tracks:
                    for msg in track:
                        if msg.type == 'note_on':
                            note_value = msg.note
                            note_duration = msg.time / midi.ticks_per_beat  # Convert ticks to beats
                            notes.append(note_value)
                            durations.append(note_duration)

                # Quantize timing to the specified time grid
                quantized_notes = self.quantize_timing(zip(np.cumsum(durations), notes, durations))

                # Map notes to integers
                for note in quantized_notes:
                    if note[1] not in self.note_to_int:
                        self.note_to_int[note[1]] = len(self.note_to_int)

                self.all_sequences.append(quantized_notes)
                self.all_tempos.append(tempo)

        return self.all_sequences, self.all_tempos

    def create_input_sequences(self, sequence_length):
        input_sequences = []

        for sequence in self.all_sequences:
            for i in range(0, len(sequence) - sequence_length):
                input_seq = sequence[i:i + sequence_length]
                input_sequences.append(input_seq)

        return input_sequences

    def normalize_data(self, input_sequences):
        scaler = StandardScaler()
        scaler.fit(np.array(input_sequences).reshape(-1, 3))
        return scaler

    def split_data(self, input_sequences, tempos, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(input_sequences, tempos, test_size=test_size,
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test
