from scipy.stats import beta
import numpy as np
import random

RANGES = [(1, 6), (7, 16), (17, 21)]
AVERAGES = [4, 11, 18]

def simulate_agents(num_agents, num_problems, okay_E, good_E, T, O, I):
    for i in range(num_agents):
        level = random.choice(range(3)) # Each agent is a random level.
        num_correct = 0
        for p in range(num_problems):
            interval = random.choice(range(25)) # Each problem is a random interval.
            (a, b) = I[level][interval]
            E = a / (a + b) # This is the frequency with which we expect to be correct.
            result = np.array(beta.rvs(a, b, size=1))
            num_correct += (result[0] > E)
            if (result[0] > E):
                O[interval][level][1] += 1
            else:
                O[interval][level][0] += 1

        # Now we've got the number of correct answers, so we're going to figure out our new states.
        num_incorrect = num_problems - num_correct

        sp = -1
        if ((num_correct + 1) / (num_problems + 2) >= good_E):
            sp = 2
        elif ((num_correct + 1) / (num_problems + 2) >= okay_E):
            sp = 1
        else:
            sp = 0  # The state we move to depends on the number of correct answers during the test.
        T[level][interval][sp] += 1
        # print("Agent: " + str(i))
        # print("Model: " + str(level))
        # print("Transitions to: " + str(sp))
        # print("---------------------------")
    state_sums = T.sum(axis=2, keepdims=True)
    o_sums = O.sum(axis=2, keepdims=True)
    T /= state_sums
    O /= o_sums
    return T, O

def create_I(num_levels, num_intervals, num_questions):
    levels = range(num_levels)
    intervals = range(num_intervals)

    I = {
        level: {interval: (0, 0) for interval in intervals}
        for level in levels
    }

    for level in levels:
        for interval in intervals:
            a = AVERAGES[level]
            I[level][interval] = (a, num_questions + 2 - a)
    return I

def run_simulation(num_levels, num_intervals, num_agents, num_problems):
    I = create_I(3, 25, num_problems)

    good_cutoff = RANGES[2][0] / (num_problems + 2)
    okay_cutoff = RANGES[1][0] / (num_problems + 2)

    T, O = simulate_agents(num_agents, num_problems, okay_cutoff, good_cutoff, np.ones((num_levels, num_intervals, num_levels)), np.ones((num_intervals, num_levels, 2)), I)
    np.set_printoptions(precision=5, suppress=True)
    # print(T)
    # print(T.shape)
    # print("---------------------------")
    return T


def main():
    num_agents = 10000
    num_problems = 20

    run_simulation(3, 25, num_agents, num_problems)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/