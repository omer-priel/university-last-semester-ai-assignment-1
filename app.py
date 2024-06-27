import numpy as np
import matplotlib.pyplot as plt


INPUT_PATH = 'two_circle.txt'


def read_input() -> np.ndarray:
    inputs = []
    with open(INPUT_PATH, 'r') as f:
        for line in f.readlines():
            x, y, label = line.strip().split()
            inputs.append((float(x), float(y), int(label)))    
    return np.array(inputs)


def train(
    inputs: np.ndarray,
    g: float,
    w: np.ndarray,
    max_rounds: int
) -> tuple[np.ndarray, int]:
    g = g / 2
    for rounds in range(max_rounds):
        found_mistake = False
        if (rounds - 1) % 10000 == 0:
            print(f'train {rounds}', g, w)

        norm = g * np.sqrt(np.sum(w**2))

        for i in range(inputs.shape[0]):
            if inputs[i, 2] * np.dot(inputs[i, :2], w) <= norm:
                w += inputs[i, 2] * inputs[i, :2]
                found_mistake = True
                break

        if not found_mistake:
            return w, rounds
    return w, max_rounds


def check(value: np.ndarray, w: np.ndarray) -> int:
    return 1 if np.dot(value, w) > 0 else -1


def test(inputs: np.ndarray, w: np.ndarray) -> tuple[int, int]:
    correct = 0
    mistake = 0
    for i in range(inputs.shape[0]):
        if check(inputs[i, :2], w) == inputs[i, 2]:
            correct += 1
        else:
            mistake += 1
    return correct, mistake


def main() -> None:
    inputs = read_input()

    g = 0.021
    w, rounds = train(inputs, g, np.zeros(2), 10**7)
    correct, mistake = test(inputs, w)
    print(f'g: {g}')
    print(f'Final weight: {w}')
    print(f'Number of rounds: {rounds}')
    print(f'Correct: {correct}, Mistake: {mistake}')

    labeld = [
        value[2] * 2 if check(value[:2], w) == value[2] else value[2]
        for value in inputs
    ]

    plt.scatter(inputs[:, 0], inputs[:, 1], c=labeld)

    x = np.linspace(-1, 1, 100)
    y = -w[0] / w[1] * x
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()


# g: 0.021
# Final weight: [-18.11  18.24]
# Number of rounds: 1153
# Correct: 150, Mistake: 0
