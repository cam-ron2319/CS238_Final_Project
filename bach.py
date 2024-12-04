import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from midi_tools import MidiProcessor, events_to_lstm_input


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Only the last output is used for prediction
        return out


def prepare_lstm_data(events, sequence_length=32, num_notes=89, num_durations=32):
    X, y = events_to_lstm_input(events, seq_len=sequence_length, num_notes=num_notes, num_durations=num_durations)
    return X, y


def train_lstm_model(X_tensor, y_tensor, model, criterion, optimizer, epochs=100):
    if os.path.exists("lstm_model.pth"):
        model.load_state_dict(torch.load("lstm_model.pth"))
        print("Model loaded successfully!")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(len(X_tensor)):
            # Get the current sequence and label
            batch_x = X_tensor[i:i + 1]
            batch_y = y_tensor[i:i + 1]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), "lstm_model.pth")
    print("Model saved successfully!")


def predict_sequence(sequence):
    model = LSTMModel(104, 128, 104)
    model.load_state_dict(torch.load("lstm_model.pth"))
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No gradient computation needed
        output = model(sequence)
        predicted_pitch = output[:, :89]
        predicted_duration = output[:, 89:]

        # Apply softmax to get probabilities
        predicted_pitch_probs = F.softmax(predicted_pitch, dim=-1).cpu().numpy()
        predicted_duration_probs = F.softmax(predicted_duration, dim=-1).cpu().numpy()

    return predicted_pitch_probs, predicted_duration_probs


def main():
    # Initialize the MidiProcessor with the path to your MIDI file
    midi_processor = MidiProcessor()

    # Process the MIDI file to get the events (start, end, note)
    X, y = midi_processor.process_for_lstm("/Users/cameroncamp/Downloads/bachsformer-main-2/data/midi/988-v01.mid")
    # Hyperparameters
    num_notes = 89  # 88 notes + pause
    num_durations = 15  # Define how many possible durations there are
    hidden_size = 128
    output_size = num_notes + num_durations  # Output will be note + duration
    input_size = num_notes + num_durations  # Input size is also note + duration

    # Convert data to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Initialize the model
    model = LSTMModel(input_size, hidden_size, output_size)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_lstm_model(X_tensor, y_tensor, model, criterion, optimizer, epochs=10)

    # Example model use
    predicted_pitch_probs, predicted_duration_probs = predict_sequence(X_tensor[0:2])  # Example
    print("Predicted pitch probabilities:", predicted_pitch_probs)
    print("Predicted duration probabilities:", predicted_duration_probs)


if __name__ == "__main__":
    main()
