import numpy as np
import matplotlib.pyplot as plt


def plot_2d_function(
        x: np.array,
        y: np.array,
        f_value: np.array
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.grid(True, which="both")
    ax.plot_surface(
        x, y, f_value, color="lime", label='func', linewidth=5
    )
    plt.show()

def plot_1d_function(
        x: np.array,
        f_value: np.array
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot()
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, which='both')
    ax.plot(x, f_value, color="lime", label='func', linewidth=5)
    plt.show()
