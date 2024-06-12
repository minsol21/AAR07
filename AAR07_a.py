import numpy as np
import matplotlib.pyplot as plt

# Constants
b = 1.0  # Axis length
vl = 1.0  # Velocity of the left wheel
vr = 1.0  # Velocity of the right wheel
time_step = 0.1  # Time step
num_steps = 100  # Number of steps for the simulation
num_runs = 100  # Number of runs to average over

# Function to simulate the robot's movement and compute odometry
def simulate_odometry(num_runs, num_steps, vl, vr, time_step, b):
    final_positions_true = []
    final_positions_odometry = []

    for run in range(num_runs):
        x_true, y_true, theta_true = 0.0, 0.0, 0.0
        x_odometry, y_odometry, theta_odometry = 0.0, 0.0, 0.0

        for step in range(num_steps):
            # Kinematic model
            delta_x_true = (np.cos(theta_true) / 2) * (vr + vl) * time_step
            delta_y_true = (np.sin(theta_true) / 2) * (vr + vl) * time_step
            delta_theta_true = (vr - vl) / b * time_step

            # Update the true position
            x_true += delta_x_true
            y_true += delta_y_true
            theta_true += delta_theta_true

            # Odometry update
            delta_d_left = vl * time_step
            delta_d_right = vr * time_step
            delta_x_odometry = ((delta_d_left + delta_d_right) / 2) * np.cos(theta_odometry)
            delta_y_odometry = ((delta_d_left + delta_d_right) / 2) * np.sin(theta_odometry)
            delta_theta_odometry = (delta_d_right - delta_d_left) / b

            # Update the odometry position
            x_odometry += delta_x_odometry
            y_odometry += delta_y_odometry
            theta_odometry += delta_theta_odometry

        final_positions_true.append((x_true, y_true))
        final_positions_odometry.append((x_odometry, y_odometry))

    return np.array(final_positions_true), np.array(final_positions_odometry)

# Function to plot histograms
def plot_histograms(final_positions_true, final_positions_odometry, title):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist2d(final_positions_true[:, 0], final_positions_true[:, 1], bins=30, cmap='Blues')
    plt.title('True Final Positions - ' + title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Count')

    plt.subplot(1, 2, 2)
    plt.hist2d(final_positions_odometry[:, 0], final_positions_odometry[:, 1], bins=30, cmap='Reds')
    plt.title('Odometry Final Positions - ' + title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Count')

    plt.tight_layout()
    plt.savefig("plot_a.png")
    plt.show()

# Simulate robot motion in a straight line along the x-axis
true_positions, odometry_positions = simulate_odometry(num_runs, num_steps, vl, vr, time_step, b)
plot_histograms(true_positions, odometry_positions, "Straight Line")
