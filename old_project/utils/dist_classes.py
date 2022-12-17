# """
# Probability distributions

# Authors
# Sam Dawley & Oliver Wolff
# """
# import numpy as np
# from mpmath import nsum
# from math import factorial
# from scipy.stats import rv_continuous, uniform

# class CMP(rv_continuous):
#     """
#     Conway-Maxwell Poisson probability distribution
#     """
#     def __init__(self, loc: float, scale: float, a: float=0, b: float=np.inf) -> None:
#         # try:
#         #     if loc <= 0 or scale <= 0:
#         #         raise ValueError("Joint posterior parameters for (lambda, nu) result in intractable solutions.")
#         # except ValueError:
#         #     for l, s in zip(loc, scale):
#         #         if l <= 0 or s <= 0:
#         #             raise ValueError("Joint posterior parameters for (lambda, nu) result in intractable solutions.")
#         self.loc = loc
#         self.scale = scale
#     def _rvs(self):
#         """
#         Random variates
#         """
#         pass
#     def _cdf(self):
#         """
#         CDF
#         """
#         pass
#     def pdf(self, z: float) -> float:
#         """
#         Probability density function
#         """
#         Q = nsum(lambda j: (self.loc**j)/(factorial(int(j))**self.scale), [0, np.inf])
#         return (1/Q) * self.loc**z / (factorial(z)**self.scale) 
#     def log_pdf(self, z: float) -> float:
#         """
#         Probability density function
#         """
#         Q = nsum(lambda j: (self.loc**j)/(factorial(int(j))**self.scale), [0, np.inf])
#         return z*np.log(self.loc) - self.scale*np.log(float(factorial(z))) - np.log(Q)

# class CMP_posterior(rv_continuous):
#     """
#     Joint posterior distribution for (lambda, nu) which is conjugate for the Conway-Maxwell Poisson distribution
#     """
#     def __init__(self, y: int, a: float, b: float, w: float) -> None:
#         if w <= 0:
#             raise ValueError("Joint posterior parameters for (lambda, nu) result in intractable solutions.")
#         self.y = y # observed sample point
#         self.a = a
#         self.b = b
#         self.w = w # 0 < w < infty
#     def _factorial_logarithm(self):
#         try:
#             return np.log(float(factorial(self.y)))
#         except OverflowError: # Use Stirling's approximation in the case that y! is intractable
#             return self.y*np.log(self.y) - self.y
#     def pdf(self, loc: float, scale: float) -> float:
#         """
#         Joint probability density
#         Returns probability of observing (lambda, nu)
#         Note that lambda = loc, nu = scale
#         """
#         numerator = (self.y+self.a) * (self.b+self._factorial_logarithm()) * loc**(self.y+self.a-1) * np.exp(-scale*(self.b+self._factorial_logarithm()))
#         return numerator / (self.w**(self.y+self.a))
#     def _stepping_out(self, x0: float, spec: str, other_param: float, weight: float=0.005, m: int=1) -> tuple:
#         """
#         Stepping-out procedure for generating an interval about a point x0 for slice sampling
#         P
#         -
#         :parama x0: initial value
#         :param spec: the variable being considered
#         :param other_param: value of other variable under within density
#         :param vert: vertical level defining the slice
#         :param weight: estimate of the typical size of a slice
#         :param m: integer limiting the size of the slice to m*w
#         """
#         U, V = uniform().rvs(2)
#         L = x0 - weight*U; R = L + weight
#         J = np.floor(m*V); K = m - 1 - J
#         # 
#         if spec == "loc":
#             while J > 0 and self.pdf(loc=x0, scale=other_param) < self.pdf(loc=L, scale=other_param):
#                 L -= weight; J -= 1
#             while K > 0 and self.pdf(loc=x0, scale=other_param) < self.pdf(loc=L, scale=other_param):
#                 R += weight; K -= 1
#             return (L, R)
#         else:
#             while J > 0 and self.pdf(scale=x0, loc=other_param) < self.pdf(scale=L, loc=other_param):
#                 L -= weight; J -= 1
#             while K > 0 and self.pdf(scale=x0, loc=other_param) < self.pdf(scale=L, loc=other_param):
#                 R += weight; K -= 1
#             return (L, R)
#     def _shrinkage(self, x0: float, spec: str, other_param: float, interval: tuple) -> float:
#         """
#         Shrinkage procedure for generating a sample from the interlva produced in self._stepping_out()
#         P
#         -
#         :parama x0: initial value
#         :param spec: the variable being considered
#         :param other_param: value of other variable under within density
#         :param vert: vertical level defining the slice
#         :param interval: tuple of (L, R) defining interval to sample from
#         """
#         Lhat, Rhat = interval
#         # Check for determing which parameter is being updated
#         if spec == "loc":
#             while True:
#                 U = uniform().rvs()
#                 x1 = Lhat + U * (Rhat - Lhat)
#                 # print(x1)
#                 if self.pdf(loc=x0, scale=other_param) < self.pdf(loc=x1, scale=other_param): 
#                     return x1
#                 if x1 < x0:
#                     Lhat = x1
#                 else:
#                     Rhat = x1
#         else:
#             while True:
#                 U = uniform().rvs()
#                 x1 = Lhat + U * (Rhat - Lhat)
#                 # print(x1)
#                 if self.pdf(loc=other_param, scale=x0) < self.pdf(loc=other_param, scale=x1): 
#                     return x1
#                 if x1 < x0:
#                     Lhat = x1
#                 else:
#                     Rhat = x1

#     def rvs(self, initial_loc: float=1000, initial_scale: float=20, N: int=1) -> np.array:
#         """
#         Slice-sampling method for generating random variates
#         """
#         loc0, scale0 = initial_loc, initial_scale
#         locs, scales = np.zeros(N), np.zeros(N)
#         for i in range(N):
#             interval = self._stepping_out(loc0, "loc", scale0)
#             loc0 = self._shrinkage(loc0, "loc", scale0, interval)
#             interval = self._stepping_out(scale0, "scale", loc0)
#             scale0 = self._shrinkage(scale0, "scale", loc0, interval)
#             locs[i] += loc0; scales[i] += scale0
#         return locs, scales

# if __name__ == "__main__":
#     d = CMP_posterior(100, 0.1, 0.1, 0.1)
#     l, s = d.rvs(N=100)
#     print(s)