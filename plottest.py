import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10)
y_sin = np.sin(x)
y_cos = np.cos(x)

p1, = plt.plot(x, y_sin)
p2, = plt.plot(x, y_cos)
plt.legend([p1, p2], ["sin", "cos"])
plt.show()
