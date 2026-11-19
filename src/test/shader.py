import numpy as np
import matplotlib.pyplot as plt


width, height = 800, 600
x, y = np.meshgrid(np.arange(width), np.arange(height))

# Apply fract(z) = z - floor(z), as in the shader expression.
image = (52.9829189 * (0.06711056 * x + 0.00583715 * y) % 1.0) % 1.0
# image = (52.9829189 * ((0.06711056 * x + 0.00583715 * y) % 1.0)) % 1.0

plt.imshow(image, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
plt.title("fract(52.9829189 * fract(0.06711056*x + 0.00583715*y))")
plt.axis("off")
plt.show()

# ACES filmic tone-mapping curve.
aces_x = np.linspace(0.0, 3.0, 1000)
aces = (aces_x * (2.51 * aces_x + 0.03)) / (
    aces_x * (2.43 * aces_x + 0.59) + 0.14
)

plt.plot(aces_x, aces)
plt.title("ACES Filmic Tone-Mapping Curve")
plt.xlabel("Input exposure")
plt.ylabel("Mapped value")
plt.xlim(0.0, 3.0)
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.show()
