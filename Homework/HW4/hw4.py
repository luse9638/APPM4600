######################################################################## imports

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

############################################################################# 1)
################################### a)

# constants
alpha = 0.138 * 10 ** -6
t_f = 5184000
T_i = 20
T_s = -15
# 500 x-values, x between 0 and 1
x = np.linspace(0, 1, 500)
# y = f(x) = erf(x / 2sqrt(alpha * t_f)) + (T_s) / (T_i - T_s)
y = sp.special.erf((x) / (2 * np.sqrt(alpha * t_f))) + (((T_s)) / (T_i - T_s))
# y0 = 0
y0 = np.zeros([len(x)])
# plot!
plt.plot(x, y)
plt.plot(x, y0)
plt.title("Problem 1a)")
plt.show()
