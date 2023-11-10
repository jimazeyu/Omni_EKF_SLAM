import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse

# Constants
NP_RANDOM_SEED = 0
LANDMARKS = 1.2 * np.array([
    [10.0, 5.0], [5.0, 10.0], [-5.0, 10.0], [-10.0, 5.0],
    [10.0, -5.0], [5.0, -10.0], [-5.0, -10.0], [-10.0, -5.0]
])
OBSERVATION_RANGE = 20.0
OBSERVATION_ANGLE = np.deg2rad(45.0)
R = np.diag([0.1**2, 0.08**2, np.deg2rad(5.0)**2])
Q = np.diag([0.1**2, np.deg2rad(5.0)**2])

# Set random seed for reproducibility
np.random.seed(NP_RANDOM_SEED)

def move(true_pos, control_input):
    """
    Moves the true position based on the control input.

    Parameters:
    true_pos (numpy.ndarray): The current true position.
    control_input (numpy.ndarray): The control input.

    Returns:
    numpy.ndarray: The new true position.
    """
    true_pos[0] += control_input[0] * np.cos(true_pos[2]) - control_input[1] * np.sin(true_pos[2]) + np.random.randn() * R[0, 0]
    true_pos[1] += control_input[0] * np.sin(true_pos[2]) + control_input[1] * np.cos(true_pos[2]) + np.random.randn() * R[1, 1]
    true_pos[2] += control_input[2] + np.random.randn() * R[2, 2]
    # Ensure theta is within -pi to pi
    true_pos[2] = np.mod(true_pos[2] + np.pi, 2 * np.pi) - np.pi
    return true_pos

def observe(true_pos, landmarks):
    """
    Observes the landmarks based on the current true position.

    Parameters:
    true_pos (numpy.ndarray): The current true position.
    landmarks (numpy.ndarray): The positions of landmarks.

    Returns:
    list: A list of observed landmarks with distance, angle, and index.
    """
    observations = []
    for index, landmark in enumerate(landmarks):
        relative_angle = true_pos[2] - np.arctan2(landmark[1] - true_pos[1], landmark[0] - true_pos[0])
        if np.abs(relative_angle) < OBSERVATION_ANGLE:
            dx, dy = landmark[0] - true_pos[0], landmark[1] - true_pos[1]
            distance = np.sqrt(dx**2 + dy**2) + np.random.randn() * Q[0, 0]
            angle = np.arctan2(dy, dx) - true_pos[2] + np.random.randn() * Q[1, 1]
            if distance <= OBSERVATION_RANGE:
                observations.append([distance, angle, index])
    return observations

def plot_covariance_ellipse(state_est, cov_est, color='red'):
    cov_xy = cov_est[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(cov_xy)
    largest_idx = np.argmax(eigval)
    smallest_idx = 1 - largest_idx

    a = np.sqrt(eigval[largest_idx])
    b = np.sqrt(eigval[smallest_idx])
    angle = np.arctan2(eigvec[largest_idx, 1], eigvec[largest_idx, 0])
    ell = Ellipse(xy=(state_est[0], state_est[1]),
                  width=a * 10, height=b * 10,
                  angle=np.rad2deg(angle),
                  color=color, fill=False)
    return ell

# Initialize true state and estimated state
true_state = np.array([0.0, -5.0, 0.0])
state_est_full = np.array([0.0, -5.0, 0.0])
cov_est_full = np.diag([0.0, 0.0, 0.0])

# Initialize plot
fig, ax = plt.subplots()
plt.grid(True)
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_aspect('equal')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
true_line, = ax.plot(true_state[0], true_state[1], 'bo', label='True Position')
est_line, = ax.plot(state_est_full[0], state_est_full[1], 'ro', label='Estimated Position')
ax.legend()

# Initialize variables for animation
cov_ellipse = []
cov_ellipse_landmarks = []
true_landmarks = []
seen_landmarks = []
observed_landmarks = []
landmarks_id = []

# Animation update function
def update(frame):
    global true_state, state_est_full, cov_est_full
    global cov_ellipse, true_landmarks, seen_landmarks, observed_landmarks, cov_ellipse_landmarks, landmarks_id

    # Simulate control input (move 0.1m in x, rotate 1° in theta)
    control_input = np.array([0.10, 0.0, np.deg2rad(1.0)])
    
    # Simulate true state
    true_state = move(true_state, control_input)

    # Obtain observations
    observations = observe(true_state, LANDMARKS)
    
    # Process each observation
    for observation in observations:
        landmark_id = int(observation[2])
        # Check if landmark is new
        if landmark_id not in landmarks_id:
            landmarks_id.append(landmark_id)
            # Initialize landmark position
            x1 = state_est_full[0] + observation[0] * np.cos(state_est_full[2] + observation[1])
            x2 = state_est_full[1] + observation[0] * np.sin(state_est_full[2] + observation[1])
            # Extend state_est_full and cov_est_full
            state_est_full = np.hstack((state_est_full, np.array([x1, x2])))
            cov_est_full_temp = np.zeros((cov_est_full.shape[0] + 2, cov_est_full.shape[1] + 2))
            cov_est_full_temp[:cov_est_full.shape[0], :cov_est_full.shape[1]] = cov_est_full
            np.fill_diagonal(cov_est_full_temp[-2:, -2:], 1e10)
            cov_est_full = cov_est_full_temp

    # Predict robot coordinates
    Fx = np.hstack((np.eye(3), np.zeros((3, 2 * len(landmarks_id)))))
    theta = state_est_full[2]
    uR = np.array([control_input[0] * np.cos(theta) - control_input[1] * np.sin(theta),
                   control_input[0] * np.sin(theta) + control_input[1] * np.cos(theta),
                   control_input[2]])
    state_est_full = state_est_full + Fx.T @ uR
    
    # Update robot covariance
    G = np.array([[0, 0, -control_input[0] * np.sin(theta) - control_input[1] * np.cos(theta)],
                  [0, 0, control_input[0] * np.cos(theta) - control_input[1] * np.sin(theta)],
                  [0, 0, 0]])
    G = Fx.T @ G @ Fx + np.eye(Fx.shape[1])
    cov_est_full = G.T @ cov_est_full @ G + Fx.T @ R @ Fx

    # Ensure theta is within -pi to pi
    state_est_full[2] = np.mod(state_est_full[2] + np.pi, 2 * np.pi) - np.pi

    # Observation correction
    for obs in observations:
        landmark_id = int(obs[2])
        # Find the index of the landmark in the state estimate
        list_id = landmarks_id.index(landmark_id)

        delta = np.array([
            state_est_full[3 + 2 * list_id] - state_est_full[0],
            state_est_full[4 + 2 * list_id] - state_est_full[1]
        ])
        q = np.dot(delta, delta)
        z_hat = np.array([
            np.sqrt(q),
            np.arctan2(delta[1], delta[0]) - state_est_full[2]
        ])

        # Normalize angle to be within -pi to pi
        z_hat[1] = np.mod(z_hat[1] + np.pi, 2 * np.pi) - np.pi

        Fxj = np.zeros((5, 3 + 2*len(landmarks_id)))
        Fxj[0:3, 0:3] = np.eye(3)
        Fxj[3:5, 2*list_id + 3:2*list_id + 5] = np.eye(2)
        H = 1 / q * np.array([
            [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], 0, np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
            [delta[1], -delta[0], -q, -delta[1], delta[0]]
        ])
        H = np.dot(H, Fxj)
        K = np.dot(cov_est_full, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(cov_est_full, H.T)) + Q)))
        state_est_full += np.dot(K, (obs[:2] - z_hat))
        cov_est_full = np.dot((np.eye(len(state_est_full)) - np.dot(K, H)), cov_est_full)

    # Update plots
    true_line.set_data(true_state[0], true_state[1])
    est_line.set_data(state_est_full[0], state_est_full[1])

    # Update landmark plots
    for landmark_plot in observed_landmarks:
        landmark_plot.remove()
    observed_landmarks = []
    for i in range(3, len(state_est_full), 2):
        lx, ly = state_est_full[i], state_est_full[i + 1]
        observed_landmarks.append(ax.add_patch(plt.Circle((lx, ly), 0.5, color='r', fill=False)))

    # Update true landmarks
    for landmark_plot in true_landmarks:
        landmark_plot.remove()
    true_landmarks = []
    for landmark in LANDMARKS:
        true_landmarks.append(ax.add_patch(plt.Circle(landmark, 0.5, color='g', fill=True)))

    # Update seen landmarks
    for landmark_plot in seen_landmarks:
        landmark_plot.remove()
    seen_landmarks = []
    for obs in observations:
        landmark_id = int(obs[2])
        landmark = LANDMARKS[landmark_id]
        seen_landmarks.append(ax.add_patch(plt.Circle(landmark, 0.5, color='b', fill=True)))

    # Update covariance ellipse for the robot
    if cov_ellipse:
        cov_ellipse.remove()
    cov_ellipse = plot_covariance_ellipse(state_est_full[:3], cov_est_full[:3, :3])
    ax.add_patch(cov_ellipse)

    # Update covariance ellipses for landmarks
    if cov_ellipse_landmarks:
        for ellipse in cov_ellipse_landmarks:
            ellipse.remove()
    cov_ellipse_landmarks = []
    for i in range(3, len(state_est_full), 2):
        ellipse = plot_covariance_ellipse(state_est_full[i:i + 2], cov_est_full[i:i + 2, i:i + 2], 'blue')
        cov_ellipse_landmarks.append(ax.add_patch(ellipse))

    # Draw the estimated position's orientation
    arrow_length = 1.0  # Length of the arrow
    est_arrow = ax.quiver(
        state_est_full[0], state_est_full[1],
        arrow_length * np.cos(state_est_full[2]), arrow_length * np.sin(state_est_full[2]),
        color='r', scale=15
    )

    # Draw the true position's orientation
    true_arrow = ax.quiver(
        true_state[0], true_state[1],
        arrow_length * np.cos(true_state[2]), arrow_length * np.sin(true_state[2]),
        color='g', scale=15
    )

    # Draw the Field of View (FOV)
    fov = ax.add_patch(plt.Polygon([
        [state_est_full[0], state_est_full[1]],
        [state_est_full[0] + 5 * np.cos(state_est_full[2] + OBSERVATION_ANGLE), state_est_full[1] + 5 * np.sin(state_est_full[2] + OBSERVATION_ANGLE)],
        [state_est_full[0] + 5 * np.cos(state_est_full[2] - OBSERVATION_ANGLE), state_est_full[1] + 5 * np.sin(state_est_full[2] - OBSERVATION_ANGLE)]
    ], color='r', fill=False))

    return true_line, est_line, cov_ellipse, *true_landmarks, *seen_landmarks, *observed_landmarks, *cov_ellipse_landmarks, est_arrow, true_arrow, fov
# Create animation
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 20, 0.1), blit=True, interval=10)

# Display plot
plt.show()
