# APPM 4600 HW 1
# Author: Luke Sellmayer, luse9638

import numpy as np
import matplotlib.pyplot as plt
import math

# Problem 1.i: plot p(x) = (x - 2)^2, using the expanded form, 
# for x between 1.920 and 2.080 (inclusive) at intervals of 0.001

# defining x values and p(x)
x = np.arange(1.920, 2.080, 0.001)
p = lambda a: (a**9) - (18 * a**8) + (144 * a**7) - (672 * a**6)\
    + (2016 * a**5) - (4032 * a**4) + (5376 * a**3) - (4608 * a**2)\
    + (2304 * a) - 512
# evaluate y1 = p(x)
y1 = p(x)
# plot x vs y1
plt.figure()
plt.scatter(x, y1, s = 10)


# Problem 1.ii: now plotting p(x) = (x - 2)^2 without expanding for same x
# values as above

# now define q(x) = (x - 2)^9
q = lambda a: (a - 2)**9
# evaluate y2 = q(x)
y2 = q(x)
# plot x vs y2
plt.figure()
plt.scatter(x, y2, s = 10)


# Problem 1.iii: Explain differences/discrepencies between the graphs, which
# graph is correct?

# When using the unexpanded form, p(x) = (x - 2)^2, for values of x very close
# to 2, x - 2 ~= 0, and 0^9 = 0. Thus on the graph, we see a straight line of
# y = 0 values. The expanded form using binomial coefficients does not have
# this issue, and so we see accurate y values. Therefore, the expanded
# coefficient form of the plot is the correct plot.

# Problem 5b)

# two x values
x1 = np.pi
x2 = 10**6

# array of delta values
delta = np.array([10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 1])

# functions
ogexpr = lambda a, d: np.cos(a + d) - np.cos(a)
newexpr = lambda a, d: -2 * np.sin((d / 2) + a) * np.sin((d / 2))

# original cos expression
x1ogexpr = ogexpr(x1, delta)
x2ogexpr = ogexpr(x2, delta)

# new derived expressions
x1newexpr = newexpr(x1, delta)
x2newexpr = newexpr(x2, delta)

# difference between original and derived
x1delt = np.abs(x1newexpr - x1ogexpr)
x2delt = np.abs(x2newexpr - x2ogexpr)

# plot delta v x1deltas
plt.figure()
plt.plot(delta, x1delt)
plt.xscale('log')
plt.yscale('log')

# plot delta v x2deltas
plt.figure()
plt.plot(delta, x2delt)
plt.xscale('log')
plt.yscale('log')

# Problem 5c)

# expression created from 5c)
taylorexpr = lambda a, d: (-d * np.sin(a)) - ((d**2 / 2) * np.cos(a))

# evaluate for x1 and x2
x1taylorexpr = taylorexpr(x1, delta)
x2taylorexpr = taylorexpr(x2, delta)

# calculate difference between taylor expressions and original expression
x1taylordelt = np.abs(x1taylorexpr - x1ogexpr)
x2taylordelt = np.abs(x2taylorexpr - x2ogexpr)

plt.figure()
plt.plot(delta, x1taylordelt)
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.plot(delta, x2taylordelt)
plt.xscale('log')
plt.yscale('log')
plt.show()
