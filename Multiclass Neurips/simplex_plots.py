import ternary
import numpy as np

# Load some data, tuples (x,y,z)
# points = []
# with open("sample_data/curve.txt") as handle:
#     for line in handle:
#         points.append(list(map(float, line.split(' '))))

points = []
for x in np.linspace(0, 1, 10):
    for y in np.linspace(0, 1 - x, 10):
        z = 1 - (x + y)
        points.append((x, y, z))


# points = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

## Sample trajectory plot
figure, tax = ternary.figure(scale=1.0)
figure.set_size_inches(5, 5)

tax.boundary()
tax.gridlines(multiple=0.2, color="black")
tax.set_title("Plotting of sample trajectory data\n", fontsize=10)

# Plot the data
tax.plot(points, linewidth=2.0, label="Curve")
tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f", offset=0.02)

tax.get_axes().axis('off')
tax.clear_matplotlib_ticks()
tax.legend()
tax.show()