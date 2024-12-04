import numpy as np
import mido
from mido import MidiFile

tick_map = {
    40: 0,  # Triplet 32nd note
    60: 1,  # 32nd note
    80: 2,  # Triplet 16th note
    90: 3,  # Dotted 32nd note
    120: 4,  # 16th note
    160: 5,  # Triplet 8th note
    180: 6,  # Dotted 16th
    240: 7,  # 8th note
    320: 8,  # Triplet quarter note
    360: 9,  # Dotted 8th note
    480: 10,  # Quarter note
    720: 11,   # Dotted quarter
    960: 12,  # Half note
    1080: 13,
    1920: 14,  # Whole note
}


class Midi:
    def __init__(self):
        note_map = {0: 'A', 1: 'Bb', 2: 'B', 3: 'C',
                    4: 'C#', 5: 'D', 6: 'Eb', 7: 'E',
                    8: 'F', 9: 'F#', 10: 'G', 11: 'Ab'}
        midi_map = {i + 1 + 20: f"{note_map[i % 12]}_{(i + 1 + 20) // 12}" for i in range(88)}
        midi_map[0] = 'pause'
        self.midi_map = midi_map
        self.note_map = {v: k for k, v in midi_map.items()}


def preprocess_top_stave(filepath):
    midi = MidiFile(filepath)
    track = midi.tracks[1]  # Use the first track (top stave)
    events = []
    cur_time = 0
    notes_on = []

    for msg in track:
        if msg.is_meta:
            continue
        cur_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            notes_on.append((cur_time, msg.note - 21))  # Normalize note to 0 (A0)
        elif msg.type in ['note_off', 'note_on'] and msg.velocity == 0:
            for i, (start_time, note) in enumerate(notes_on):
                if note == msg.note - 21:
                    events.append((start_time, cur_time, note))
                    notes_on.pop(i)
                    break
    events.sort()
    return events


def events_to_lstm_input(events, seq_len=32, num_notes=89, num_durations=15):
    """
    Converts events into a one-hot representation for LSTM training.

    Args:
        events: List of (start, end, note) tuples.
        seq_len: Length of each sequence.
        num_notes: Number of unique notes (88 notes + 1 pause).

    Returns:
        X: Input sequences (N x seq_len x num_notes).
        y: Output labels (N x num_notes).
    """
    sequences = []
    labels = []
    one_hot_pause = np.zeros(num_notes + num_durations)
    one_hot_pause[0] = 1  # Pause is the first entry

    one_hot_notes = [np.zeros(num_notes) for _ in range(88)]
    for i in range(num_notes - 1):
        one_hot_notes[i][i + 1] = 1  # Note indices start at 1

    # Initialize one-hot encodings for durations
    one_hot_durations = [np.zeros(num_durations) for _ in range(num_durations)]
    for i in range(num_durations - 1):
        one_hot_durations[i][i + 1] = 1  # Duration indices start at 1

    for i in range(len(events) - seq_len):
        sequence = []
        for j in range(seq_len):
            start, end, note = events[i + j]
            duration = end - start

            # Ensure duration is within bounds
            if duration <= 0:
                sequence.append(one_hot_pause)
            else:
                duration_idx = tick_map[duration]  # Cap to max duration
                note_duration = np.concatenate([one_hot_notes[note], one_hot_durations[duration_idx]])
                sequence.append(note_duration)

        next_note = events[i + seq_len][2]  # Predict the note after the sequence
        next_duration = events[i + seq_len][1] - events[i + seq_len][0]  # Duration for the next note

        # Find the corresponding index for the next note and duration
        duration_idx = tick_map[next_duration]
        label = np.concatenate([one_hot_notes[next_note], one_hot_durations[duration_idx]])

        sequences.append(sequence)
        labels.append(label)

    return np.array(sequences), np.array(labels)


class MidiProcessor(Midi):
    def __init__(self, seq_len=32):
        super().__init__()
        self.seq_len = seq_len

    def process_for_lstm(self, filepath):
        events = preprocess_top_stave(filepath)
        X, y = events_to_lstm_input(events, seq_len=self.seq_len)
        return X, y
