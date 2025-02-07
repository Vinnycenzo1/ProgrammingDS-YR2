import numpy as np
a = np.array([2,3,4])
print(a)

print(a.dtype)

b = np.array([1.2, 3.5, 5.2])
print(b.dtype)


a = np.array([1, 2, 3, 4]) #right

b = np.array([(1.5, 2, 3), (4, 5, 6)])
print(b)

c = np.array([[1, 2], [3, 4]], dtype=complex)
print(c)

np.zeros((3, 4))
print(np.zeros((3, 4)))

np.ones((2, 3, 4), dtype=np.int16)
print(np.ones((2, 3, 4)))

np.empty((2, 3))
print(np.empty((2, 3)))

np.arange(10, 30, 5)
print(np.arange(10, 30, 5))

np.arange(0, 2, 0.3)
print(np.arange(0, 2, 0.3))

from numpy import pi
np.linspace(0, 2, 9) # 9 numbers from 0 to 2
print(np.linspace(0, 2, 9))

x = np.linspace(0, 2 * pi, 100) # useful to evaluate function at lots of points
f = np.sin(x)
print(f)

a = np.array([20, 30, 40, 50])
b = np.arange(4)
print(b)

c = a - b
print(c)

print(b**2)

print(10 * np.sin(a))

print(a < 35)

A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])
print(A * B) # elementwise product

print(A @ B) # Matrix product

print(A.dot(B)) # Matrix product

rg = np.random.default_rng(1) # create instance of default random number generator
a = np.ones((2, 3), dtype=int)
b = rg.random((2, 3))
a *= 3
print(a)

b += a
print(b)

a = np.ones(3, dtype=np.int32)
b = np.linspace(0, pi, 3)
print(b.dtype.name)

c = a + b
print(c)

print(c.dtype.name)

d = np.exp(c * 1j)
print(d)

print(d.dtype.name)