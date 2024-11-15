import time
import numpy as np
from scipy.stats import entropy
import sounddevice as sd

from simulator import run_simulation

sample_rate = 44100
duration = 0.5
belief_state = np.full(25, 0.5)
alpha = np.ones(25)
beta_params = np.ones(25)
actions = list(range(25))
well_tempered_frequencies = [220 * 2 ** (n / 12) for n in range(13)]
interval_dict = {
    'P1': 0, 'm2': 1, 'M2': 2, 'm3': 3, 'M3': 4,
    'P4': 5, 'A4': 6, 'P5': 7, 'm6': 8, 'M6': 9,
    'm7': 10, 'M7': 11, 'P8': 12
}


def setup_model():
    states = list(range(3))
    actions = list(range(25))
    P_transitions = run_simulation(3, 25, 10000, 20)
    obs_probs = np.random.rand(2, len(states), len(actions))
    obs_probs /= obs_probs.sum(axis=1, keepdims=True)
    belief = np.full(len(states), 1 / len(states))
    return states, actions, P_transitions, obs_probs, belief


def update_belief(belief, action, observation, P_transitions, obs_probs, states):
    new_belief = np.zeros(len(states))
    for j in range(len(states)):
        obs_prob = obs_probs[observation, j, action]
        total_prob = sum(P_transitions[i, action, j] * belief[i] for i in range(len(states)))
        new_belief[j] = obs_prob * total_prob
    new_belief /= np.sum(new_belief)
    return new_belief


def expected_information_gain(belief, action, P_transitions, obs_probs, states):
    expected_kl = 0.0
    possible_observations = [0, 1]
    for observation in possible_observations:
        new_belief = update_belief(belief, action, observation, P_transitions, obs_probs, states)
        kl_divergence = entropy(new_belief, belief)
        obs_prob = sum(obs_probs[observation, i, action] * belief[i] for i in range(len(states)))
        expected_kl += obs_prob * kl_divergence
    return expected_kl


def select_action(belief, actions, P_transitions, obs_probs, states):
    info_gains = [expected_information_gain(belief, action, P_transitions, obs_probs, states) for action in actions]
    return np.argmax(info_gains)


def random_action():
    return np.random.choice(actions)


def evaluate_uncertainty(belief):
    return entropy(belief, np.ones(len(belief)) / len(belief))


def run_experiment(strategy, num_flashcards, states, actions, P_transitions, obs_probs, belief):
    initial_uncertainty = evaluate_uncertainty(belief)
    for _ in range(num_flashcards):
        if strategy == "info_gain":
            action = select_action(belief, actions, P_transitions, obs_probs, states)
        elif strategy == "random":
            action = random_action()
        observation = get_user_response(action)
        belief = update_belief(belief, action, observation, P_transitions, obs_probs, states)
    final_uncertainty = evaluate_uncertainty(belief)
    return initial_uncertainty, final_uncertainty


def generate_tone(frequency):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    envelope = np.exp(-3 * t) * np.sin(2 * np.pi * frequency * t)
    return envelope


def play_interval(semitone_steps):
    base_frequency = np.random.choice(well_tempered_frequencies)
    interval_frequency = base_frequency * (2 ** (semitone_steps / 12))
    base_tone = generate_tone(base_frequency)
    sd.play(base_tone, samplerate=sample_rate)
    sd.wait()
    time.sleep(0.2)
    interval_tone = generate_tone(interval_frequency)
    sd.play(interval_tone, samplerate=sample_rate)
    sd.wait()


def get_user_response(action):
    semitone_steps = action - 12
    while True:
        play_interval(semitone_steps)
        user_input = input("Identify the interval (or type 'replay' to hear it again): ")
        if user_input in interval_dict:
            correct = (interval_dict[user_input] == abs(semitone_steps))
            return int(correct)
        else:
            print("Invalid input. Please enter a valid interval name (e.g., 'M3', 'P5').")


def main():
    states, actions, P_transitions, obs_probs, belief = setup_model()
    num_flashcards = 20

    # Run experiment with information gain strategy
    info_gain_initial, info_gain_final = run_experiment(
        "info_gain", num_flashcards, states, actions, P_transitions, obs_probs, belief.copy()
    )
    print(f"Initial uncertainty (info gain): {info_gain_initial}")
    print(f"Final uncertainty (info gain): {info_gain_final}")

    # Run experiment with random action strategy
    random_initial, random_final = run_experiment(
        "random", num_flashcards, states, actions, P_transitions, obs_probs, belief.copy()
    )
    print(f"Initial uncertainty (random): {random_initial}")
    print(f"Final uncertainty (random): {random_final}")


if __name__ == "__main__":
    main()
