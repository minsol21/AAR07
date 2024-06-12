import numpy as np
import matplotlib.pyplot as plt

# Constants
b = 1.0  # Axis length
time_step = 0.1  # Time step
vr_circular = 1.0
vl_circular = 0
num_steps_circular = 64
num_runs = 100  # Number of runs to average over
sigma = 0.05  # Standard deviation of the velocity error
sigma_od_right = 0.1  # Standard deviation of odometry noise for the right wheel
sigma_od_left = 0.1   # Standard deviation of odometry noise for the left wheel

# Function to simulate the robot's movement and compute odometry
def simulate_odometry(num_runs, num_steps, vl, vr, time_step, b, sigma, sigma_od_right, sigma_od_left):
    final_positions_true = []
    final_positions_no_correction = []
    final_positions_perfect_correction = []
    final_positions_noisy_correction = []

    for run in range(num_runs):
        x_true, y_true, theta_true = 0.0, 0.0, 0.0
        x_no_correction, y_no_correction, theta_no_correction = 0.0, 0.0, 0.0
        x_perfect_correction, y_perfect_correction, theta_perfect_correction = 0.0, 0.0, 0.0
        x_noisy_correction, y_noisy_correction, theta_noisy_correction = 0.0, 0.0, 0.0

        for step in range(num_steps):
            # Introduce errors to the velocities
            epsilon_right = np.random.normal(0, sigma)
            epsilon_left = np.random.normal(0, sigma)
            vr_noisy = vr + epsilon_right
            vl_noisy = vl + epsilon_left
            
            # Kinematic model
            delta_x_true = (np.cos(theta_true) / 2) * (vr_noisy + vl_noisy) * time_step
            delta_y_true = (np.sin(theta_true) / 2) * (vr_noisy + vl_noisy) * time_step
            delta_theta_true = (vr_noisy - vl_noisy) / b * time_step

            # Update the true position
            x_true += delta_x_true
            y_true += delta_y_true
            theta_true += delta_theta_true

            # Odometry update
            delta_d_left = vl * time_step + np.random.normal(0, sigma_od_left)
            delta_d_right = vr * time_step + np.random.normal(0, sigma_od_right)
            delta_x_no_correction = ((delta_d_left + delta_d_right) / 2) * np.cos(theta_no_correction)
            delta_y_no_correction = ((delta_d_left + delta_d_right) / 2) * np.sin(theta_no_correction)
            delta_theta_no_correction = (delta_d_right - delta_d_left) / b

            # Update the position without correction
            x_no_correction += delta_x_no_correction
            y_no_correction += delta_y_no_correction
            theta_no_correction += delta_theta_no_correction

            # Odometry correction with perfect estimation
            delta_d_left_perfect = vl * time_step
            delta_d_right_perfect = vr * time_step
            delta_x_perfect_correction = ((delta_d_left_perfect + delta_d_right_perfect) / 2) * np.cos(theta_perfect_correction)
            delta_y_perfect_correction = ((delta_d_left_perfect + delta_d_right_perfect) / 2) * np.sin(theta_perfect_correction)
            delta_theta_perfect_correction = (delta_d_right_perfect - delta_d_left_perfect) / b

            # Update the position with perfect odometry correction
            x_perfect_correction += delta_x_perfect_correction
            y_perfect_correction += delta_y_perfect_correction
            theta_perfect_correction += delta_theta_perfect_correction

            # Odometry correction with noisy estimation
            delta_d_left_noisy = delta_d_left + np.random.normal(0, sigma_od_left)
            delta_d_right_noisy = delta_d_right + np.random.normal(0, sigma_od_right)
            delta_x_noisy_correction = ((delta_d_left_noisy + delta_d_right_noisy) / 2) * np.cos(theta_noisy_correction)
            delta_y_noisy_correction = ((delta_d_left_noisy + delta_d_right_noisy) / 2) * np.sin(theta_noisy_correction)
            delta_theta_noisy_correction = (delta_d_right_noisy - delta_d_left_noisy) / b

            # Update the position with noisy odometry correction
            x_noisy_correction += delta_x_noisy_correction
            y_noisy_correction += delta_y_noisy_correction
            theta_noisy_correction += delta_theta_noisy_correction

        final_positions_true.append((x_true, y_true))
        final_positions_no_correction.append((x_no_correction, y_no_correction))
        final_positions_perfect_correction.append((x_perfect_correction, y_perfect_correction))
        final_positions_noisy_correction.append((x_noisy_correction, y_noisy_correction))

    return (
        np.array(final_positions_true), 
        np.array(final_positions_no_correction), 
        np.array(final_positions_perfect_correction),
        np.array(final_positions_noisy_correction)
    )

# Function to plot histograms
def plot_histograms(final_positions_true, final_positions_no_correction, final_positions_perfect_correction, final_positions_noisy_correction, title):
    plt.figure(figsize=(20, 8))

    plt.subplot(2, 2, 1)
    plt.hist2d(final_positions_true[:, 0], final_positions_true[:, 1], bins=30, cmap='Blues')
    plt.title('True Final Positions - ' + title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Count')

    plt.subplot(2, 2, 2)
    plt.hist2d(final_positions_no_correction[:, 0], final_positions_no_correction[:, 1], bins=30, cmap='Greens')
    plt.title('No Correction Final Positions - ' + title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Count')

    plt.subplot(2, 2, 3)
    plt.hist2d(final_positions_perfect_correction[:, 0], final_positions_perfect_correction[:, 1], bins=30, cmap='Oranges')
    plt.title('Perfect Correction Final Positions - ' + title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Count')

    plt.subplot(2, 2, 4)
    plt.hist2d(final_positions_noisy_correction[:, 0], final_positions_noisy_correction[:, 1], bins=30, cmap='Reds')
    plt.title('Noisy Correction Final Positions - ' + title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Count')

    plt.tight_layout()
    plt.savefig("plot_c_"+title+".png")  
    plt.show()

# Simulate robot motion in a circular path
(
    final_positions_true, 
    final_positions_no_correction, 
    final_positions_perfect_correction, 
    final_positions_noisy_correction
) = simulate_odometry(num_runs, num_steps_circular, vl_circular, vr_circular, time_step, b, sigma, sigma_od_right, sigma_od_left)

plot_histograms(
    final_positions_true, 
    final_positions_no_correction, 
    final_positions_perfect_correction, 
    final_positions_noisy_correction, 
    "Circular Path"
)
