import matplotlib.pyplot as plt

# Data for the line plots
x = [1, 2, 3, 4, 5]
y1 = [2, 3, 5, 7, 11]
y2 = [1, 4, 9, 16, 25]

# Plot the data
plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')

# Add legend
plt.legend()

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot with Legend')

# Show the plot
plt.show()
