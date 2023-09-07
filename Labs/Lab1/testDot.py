import numpy as np
import numpy.linalg as la

def driver():

     n = 100
     x = np.linspace(0,np.pi,n)

     # orthogonal vectors
     x1 = np.array([0, 1])
     x2 = np.array([1, 0])

     # 2 x 2 matrix
     M = np.array([[1, 2], [3, 4]])
     # another vector
     b = np.array([5, 6])

# this is a function handle.  You can use it to define 
# functions instead of using a subroutine like you 
# have to in a true low level language.     
     f = lambda x: x**2 + 4*x + 2*np.exp(x)
     g = lambda x: 6*x**3 + 2*np.sin(x)

     v = f(x)
     w = g(x)

# evaluate the dot product of y and w     
     dp = dotProduct(x1,x2,2)

     print("Multiplying matrix by vector using my code: ")
     print(matrixVectorMult(M, b))
     print("Verifying with numpy: ")
     print(np.matmul(M, b))

# print the output
     print('the dot product is : ', dp)
     print("Numpy dot product is: " + str(np.dot(x1, x2)))

     return
     
def dotProduct(x,y,n):

     dp = 0.
     for j in range(n):
        dp = dp + x[j]*y[j]

     return dp  

# only works for 2 x 2 matrix and 2 x 1 column vector
def matrixVectorMult(matrix: np.ndarray, vector: np.array):
     
     output = np.array([[0], [0]])
     output[0][0] = dotProduct(matrix[0,:], vector, 2)
     output[1][0] = dotProduct(matrix[1,:], vector, 2)

     return output
     
driver()               
