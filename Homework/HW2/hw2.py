import numpy as np
import matplotlib.pyplot as plt

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
print(condA)

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
print("The sum is: " + str(S))

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

plt.figure()
plt.plot(x, y)

plt.figure()
delr = 0.05
for i in range(1, 11):
    R = i
    f = 2 + i
    # x(theta) = R(1 + delrsin(ftheta + p))cos(theta)
    xcurve = R * (1 + delr * np.sin(f * theta + p)) * np.cos(theta)
    # y(theta) = R(1 + delrsin(ftheta + p))sin(theta)
    ycurve = R * (1 + delr * np.sin(f * theta + p)) * np.sin(theta)
    plt.plot(xcurve, ycurve)


plt.show()
