######################################################################## imports
import numpy as np
from fixedpt_example_modified import fixedpt

########################################################################### 3.2)
def aitkens(sequence, seqLen, tol):
    newSequence = []
    for (val, i) in list(zip(sequence, range(seqLen))):
        newSeqVal = sequence[i] - (((sequence[i + 1] - sequence[i]) ** 2)\
                                         / (sequence[i + 2] -\
                                             (2 * sequence[i + 1]) \
                                                + sequence[i]))
        newSequence.append(newSeqVal)
        if (abs(newSequence[i] - newSequence[i - 1]) / abs(newSequence[i - 1]) < tol and i != 0):
            return (newSequence, i)
    
    return newSequence

g = lambda x: ((10) / (x + 4)) ** (1 / 2)
(xStar, err, iterations) = fixedpt(g, 1.5, 10 ** -10, 1000) # converges in 12 
                                                            # iterations
print("Fixed point approximation: " + str(xStar[iterations][0]))
print("Iterations needed: " + str(iterations))
print("Error code: " + str(err))

(xStarAit, count) = aitkens(xStar, iterations, 10 ** -10) # converges in 7 iterations
print("Iterations: " + str(count) + ", Value: " + str(xStarAit[count][0]))