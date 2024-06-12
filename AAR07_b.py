import numpy as np
import matplotlib.pyplot as plt

# Constants
b = 1.0  # Axis length
vl = 1.0  # Velocity of the left wheel
vr = 1.0  # Velocity of the right wheel
time_step = 0.1  # Time step
num_steps = 10 # Number of steps for the simulation
num_runs = 100  # Number of runs to average over
sigma = 0.1  # Standard deviation of the velocity error

# Simulate for circular path.
# If we only turn one wheel, then the robot makes a circular motion 
# with the circle’s circumference being C = 2 · b · π
# for the robot to come back to the starting point, num_steps = C / time_step ~= 64
vr_circular = 1.0
vl_circular = 0
num_steps_circular = 64

# Function to simulate the robot's movement and compute odometry
def simulate_odometry(num_runs, num_steps, vl, vr, time_step, b, sigma, circular=False):
    final_positions = []

    for run in range(num_runs):
        x, y, theta = 0.0, 0.0, 0.0
        x_od, y_od, theta_od = 0.0, 0.0, 0.0

        for step in range(num_steps):
            # Introduce errors to the velocities
            epsilon_right = np.random.normal(0, sigma)
            epsilon_left = np.random.normal(0, sigma)
            vr_noisy = vr + epsilon_right
            vl_noisy = vl + epsilon_left
            
            # Kinematic model
            delta_x = (np.cos(theta) * (vr_noisy + vl_noisy) / 2) * time_step
            delta_y = (np.sin(theta) * (vr_noisy + vl_noisy) / 2) * time_step
            delta_theta = (vr_noisy - vl_noisy) / b * time_step

            # Update the true position
            x += delta_x
            y += delta_y
            theta += delta_theta

            # Odometry update (assuming perfect odometry)
            delta_d_left = vl * time_step
            delta_d_right = vr * time_step
            delta_x_od = ((delta_d_left + delta_d_right) / 2) * np.cos(theta_od)
            delta_y_od = ((delta_d_left + delta_d_right) / 2) * np.sin(theta_od)
            delta_theta_od = (delta_d_right - delta_d_left) / b

            # Update the odometry position
            x_od += delta_x_od
            y_od += delta_y_od
            theta_od += delta_theta_od

        final_positions.append((x, y, x_od, y_od))
        true_positions = np.array([[pos[0], pos[1]] for pos in final_positions])
        odometry_positions = np.array([[pos[2], pos[3]] for pos in final_positions])

    return true_positions, odometry_positions



# Function to plot histograms
def plot_histograms(true_positions, odometry_positions, title):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist2d(true_positions[:, 0], true_positions[:, 1], bins=30, cmap='Blues')
    plt.title('True Final Positions - ' + title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Count')
    

    plt.subplot(1, 2, 2)
    plt.hist2d(odometry_positions[:, 0], odometry_positions[:, 1], bins=30, cmap='Reds')
    plt.title('Odometry Final Positions - ' + title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Count')


    plt.tight_layout()
    plt.savefig("plot_b_"+title+".png")  
    plt.show()

# Extract and plot histograms for straight line
true_positions_straight, odometry_positions_straight = simulate_odometry(num_runs, num_steps, vl, vr, time_step, b, sigma, circular=True)
plot_histograms(true_positions_straight, odometry_positions_straight, "Straight Line")





# Extract and plot histograms for circular path
true_positions_circular, odometry_positions_circular = simulate_odometry(num_runs, num_steps_circular, vl_circular, vr_circular, time_step, b, sigma, circular=True)
plot_histograms(true_positions_circular, odometry_positions_circular, "Circular Path")
