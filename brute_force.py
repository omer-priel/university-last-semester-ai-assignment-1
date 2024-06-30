import math

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


def get_vector(
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> tuple[float, float, float]:
    a = -1 * ((x2 - x1) / (y2 - y1))
    b = ((y2 + y1) / 2) - a * ((x2 + x1) / 2)
    margin = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2
    return a, b, margin


def check(a: float, b: float, x: float, y: float) -> int:
    return 1 if a * x + b < y else -1


def find(
    inputs: np.ndarray
) -> tuple[float, float, float, tuple[tuple[float, float], tuple[float, float]]]:
    group_1 = [(value[0], value[1]) for value in inputs if value[2] == -1]
    group_2 = [(value[0], value[1]) for value in inputs if value[2] == 1]

    min_value = min(
        [value[0] for value in inputs]
        + [value[1] for value in inputs]
    )
    max_value = max(
        [value[0] for value in inputs]
        + [value[1] for value in inputs]
    )

    a, b = 0, 0
    margin = ((max_value - min_value) ** 2) + 1
    selected_points = ((0.0, 0.0), (0.0, 0.0))

    for p1 in group_1:
        for p2 in group_2:
            local_a, local_b, local_margin = get_vector(p1[0], p1[1], p2[0], p2[1])
            if local_margin < margin:
                good_vector = True
                for value in inputs:
                    x, y, label = value[0], value[1], value[2]
                    if check(local_a, local_b, x, y) != label:
                        good_vector = False
                        break
                
                if good_vector:
                    a, b, margin = local_a, local_b, local_margin
                    selected_points = (p1, p2)

    return a, b, margin, selected_points


def main() -> None:
    inputs = read_input()

    a, b, margin, points = find(inputs)
    p1, p2 = points
    print(f'vector: {a}x + {b}')
    print(f'margin: {margin}')
    print(f'points: {p1}, {p2}')

    plt.scatter(inputs[:, 0], inputs[:, 1], c=inputs[:, 2])
    plt.scatter([p1[0], p2[0]], [p1[1], p2[1]], c='red')
    x = np.linspace(-1, 1, 100)
    y = a * x + b
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()


# vector: 1.0000000000000038 * x + -0.02000000000000257
# margin: 0.021213203435596406
# points: (0.73, 0.68), (0.7, 0.71)

# how does it compare to the margin that Perceptron discovered? is more accurate with delta of 0.000213203435596406
