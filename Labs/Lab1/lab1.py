import numpy as np
import matplotlib.pyplot as plt

# 3.2 exercises
x = np.linspace(0, 100, 5)
y = np.arange(0, 100, 20)

print("The first three entries of x are: " + str(x[0:3]))

# creating w values
w = 10 ** (-np.linspace(1, 10, 10))
print(w)
x = np.arange(1, len(w) + 1, 1)
print(x)
# semilogx/y use a log scale on that respective axis
plt.semilogy(x, w)
s = 3 * w
plt.semilogy(x, s)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
