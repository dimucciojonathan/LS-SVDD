import numpy as np
import pandas as pd
import time

def kernel(x, y, sigma):
    dist = np.linalg.norm(x - y) ** 2
    return np.exp(- dist / sigma)


def H_Matrix(data, C, sigma):  # From exam 1
    length = len(data)
    kmat = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            kmat[i, j] = kernel(data[i], data[j], sigma)
    # np.fill_diagonal(kmat, kmat.diagonal() + 1 / ( 2*C))
    kmat = kmat + np.eye(len(data)) * (1 / (2 * C))
    return kmat
    
nums = {}
def iterate_hn_inv(n, data, c, sigma):
    if n in nums:
        return nums[n]
    else:
        if n == 1:
            nums[n] = np.array([(2 * c) / (1 + 2 * c)])  # memoize
            return np.array([(2 * c) / (1 + 2 * c)])
        else:
            # Define variables from formula (Hn^-1)
            Sn = []
            for i in range(n - 1):
                Sn = np.append(Sn, [kernel(data[n - 1], data[i], sigma)], axis=0)
            hno = iterate_hn_inv(n - 1, data, c, sigma)  # Recursive call for Hinverse(n-1)
            Snn = kernel(data[n], data[n], sigma)
            an = np.array(np.matmul(hno, Sn))
            if n == 2:
                an = np.array([an])  # an is a scalar at n=2. It must be an array for the next line
            Yn = Snn + (1 / (2 * c)) - np.matmul(np.transpose(Sn), an)

            # Construct (nxn) Matrix
            nth_col = np.append(-an, [1], axis=0).reshape((n, 1))
            hnrow = np.vstack((Yn * hno, -an))
            hnfinal = (1 / Yn) * np.hstack((hnrow, nth_col))
            nums[n] = hnfinal  # memoize
            return hnfinal


# Part B, same function but H matrix has already been inverted
def alpha_new(H, data, C, sigma):  # H is now already inverted
    k = []
    for i in range(len(H)):
        k.append(kernel(data[i], data[i], sigma))
    e = np.ones(len(H))
    et = np.transpose(e)

    num = 2 - np.matmul(np.matmul(et, H), k)
    den = np.matmul(np.matmul(et, H), e)
    b = (num / den) * e
    alpha = .5 * np.matmul(H, k + b)
    return alpha
    
def zdistance_new(n, data, z, C, sigma):
    H = iterate_hn_inv(n, data, sigma, C)
    a = alpha_new(H, data, sigma, C)
    test = 0
    j_sum = 0
    jl_sum = 0

    for j in range(len(H)):
        j_sum += a[j] * kernel(z, data[j], sigma)
        for l in range(len(H)):
            jl_sum += a[j] * a[l] * kernel(data[j], data[l], sigma)
    test = (1 - 2 * j_sum + jl_sum)
    return test
