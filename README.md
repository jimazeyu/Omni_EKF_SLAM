# Omni_EKF_SLAM
An EKF_SLAM based on omnidirectional motion, expanding the state and covariance matrix as new landmarks are detected. Replace the move function with the actual robot's control function, replace observe with the observation function, and return a list of tuples for all landmarks (distance, angle, id), to apply the algorithm to a real robot.

![plot](plot.gif)
