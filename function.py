import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def model_func(x, A, B, C, D, E):
    return A*np.exp(B*x) + C * x**2 + D * x + E

x_data = [51.616, 68.026, 82.164, 96.292, 122.715, 139.119, 151.871, 165.077, 191.026, 206.511, 221.992, 235.192, 262.961, 276.608, 289.362, 302.556, 332.609, 347.177, 359.933, 378.61]
x_data = np.array(x_data)
y_data = [187.887, 187.893, 188.988, 188.56, 185.824, 184.719, 182.502, 179.715, 172.42, 169.011, 164.455, 159.316, 147.802, 139.149, 134.448, 125.137, 108.584, 97.921, 90.706, 78.652]

initial_guess = [1, 0.1, 1, 1, 1]

popt, pcov = curve_fit(model_func, x_data, y_data, p0=initial_guess)

A_fit, B_fit, C_fit, D_fit, E_fit = popt
print("Fitted parameters:")
print(f"A = {A_fit}, B = {B_fit}, C = {C_fit}, D = {D_fit}, E = {E_fit}")

y_fit = model_func(x_data, *popt)

plt.scatter(x_data, y_data, label='Data', color='gray')
plt.plot(x_data, y_fit, label='Fitted curve', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Nonlinear Fit: $y = Ae^{Bx} + Cx^2 + Dx + E$')
plt.grid(True)
plt.show()
