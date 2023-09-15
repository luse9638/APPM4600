import numpy as np
import matplotlib.pyplot as plt
import random
import math

############################################################################ 2)

######################################## b)

# create matrices A and A^-1
A = 0.5 * np.array([[1, 1], [1 + 10 ** -10, 1 - 10 ** -10]])
Ainv = np.array([[1 - 10 ** 10, 10 ** 10], [1 + 10 ** 10, -1 * 10 ** 10]])

# calculate maximum and minimum singular values of A
maxSVDA = np.linalg.svd(A)[1][0] # sigma_1
minSVDA = np.linalg.svd(A)[1][1] # sigma_n

# condition number, k(A) = sigma_1 / sigma_n
condA = maxSVDA / minSVDA
print("Problem 2b: condition number of matrix A: " + str(condA))

########################################## c)

delB = np.array([5 * 10 ** -5, 5 * 10 ** -5])
delX = np.matmul(Ainv, delB)
normDelX = np.linalg.norm(delX)
print("Problem 2c: Norm of delX: " + str(normDelX))
relativeError = condA * normDelX
print("Problem 2c: relative error of solutoin: " + str(relativeError))

############################################################################ 3)

############################################# c)

# algorithm to compute f(x) = e^x - 1
def algo(x):
    y = (math.e) ** x
    return y - 1

x1 = np.longdouble(9.999999995000000 * 10 ** -10)
print("Problem 3c: f(9.999999995000000 * 10^-10) = " + str(algo(x1)))

############################################# d)

# function to computer relative error with g_n(x)
def calcRelError():
    # make initial relative error really really big
    relError = np.longdouble(10 ** 10)
    # actual value, f(x1) = 10^-9
    actual = np.longdouble(10 ** -9)
    # accuracy of... a lot of  digits
    tolerance = np.longdouble(10 ** -16)
    # initial approximation is first term of Taylor Series
    g_n = np.longdouble(0)
    # iterator
    j = 0
    # continue adding terms to Taylor Series, g_n(x), approximating 
    # f(x) = e^(x) - 1 until relative error between g_n(x1) and f(x1) is below
    # tolerance
    while (relError > tolerance):
        # iterate
        j += 1
        # recalculate g_n(x1) with an additional term 
        g_n += np.longdouble((x1 ** j)) / np.longdouble((math.factorial(j)))

        # recalculate relative error
        relError = np.absolute(actual - g_n) / actual

    print("Problem 3d: terms needed: " + str(j))
    return relError

relErrorx1 = calcRelError() # says only first two terms are needed
# g_2(x) approximates f(x) using only two terms
g_2 = lambda x: x + ((x ** 2) / 2)

############################################## e)

# output values of g_2(x1) and the relative error calculated from calRelError()
# formatted to 18 decimal places of precision
print("Problem 3e: g_2(x1) = " + "{0:.40f}".format(g_2(np.longdouble(x1))))
print("Problem 3e: relative error is: " + "{0:.40f}".format(relErrorx1))

############################################################################ 4)

######################################### a)

# vector, entries from a = 0 to b = pi, incremented by pi / 30
a = 0
b = np.pi
t = np.linspace(a, b, 31) # 31 values results in step size of pi / 30
# vector y(t) = cos(t)
y = np.cos(t)
# print("PROBLEM 4 STARTS HERE")
# print("T values: " + str(t))
# print("Y values: " + str(y))

# evaulating the sum, S, from k = 1 to N of t(k) * y(k)
S = 0
for tval, yval in zip(t, y):
    # print(str(tval) + ", " + str(yval))
    S += tval * yval

# output
print("Problem 4a: The sum is: " + str(S))

######################################### b)

# 1,000 theta values between 0 and 2pi
theta = np.linspace(0, 2 * np.pi, 1000)
# parameters
R = 1.2
delr = 0.1
f = 15
p = 0
# x(theta) = R(1 + delrsin(ftheta + p))cos(theta)
x = R * (1 + delr * np.sin(f * theta + p)) * np.cos(theta)
# y(theta) = R(1 + delrsin(ftheta + p))sin(theta)
y = R * (1 + delr * np.sin(f * theta + p)) * np.sin(theta)

# plot x(theta) vs y(theta)
plt.figure()
plt.plot(x, y)
plt.title("R = 1.2, delr = 0.1, f = 15, p = 0")

# plot 10 curves with variable values of R, f, and p, delr = 0.05
plt.figure()
delr = 0.05
for i in range(1, 11):
    R = i
    f = 2 + i
    # generate uniformly distributed random number in [0, 2]
    p = random.uniform(0, 2)
    # x(theta) = R(1 + delrsin(ftheta + p))cos(theta)
    xcurve = R * (1 + delr * np.sin(f * theta + p)) * np.cos(theta)
    # y(theta) = R(1 + delrsin(ftheta + p))sin(theta)
    ycurve = R * (1 + delr * np.sin(f * theta + p)) * np.sin(theta)
    plt.plot(xcurve, ycurve)

plt.title("R = i, delr = 0.05, f = 2 + i, p between 0 and 2")
# show the pretty plots :)
plt.show()
