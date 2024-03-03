import numpy as np

class KalmanFilter2D:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, dt):
        # State prediction
        F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]])
        self.state = np.dot(F, self.state)

        # Covariance prediction
        Q = np.array([[dt**4/4, dt**3/2, 0, 0],
                      [dt**3/2, dt**2, 0, 0],
                      [0, 0, dt**4/4, dt**3/2],
                      [0, 0, dt**3/2, dt**2]]) * self.process_noise
        self.covariance = np.dot(np.dot(F, self.covariance), F.T) + Q

    def update(self, measurement):
        # Measurement update
        H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])
        R = np.eye(2) * self.measurement_noise

        y = measurement - np.dot(H, self.state)
        S = np.dot(np.dot(H, self.covariance), H.T) + R
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))

        # Update state and covariance
        self.state = self.state + np.dot(K, y)
        self.covariance = np.dot(np.eye(4) - np.dot(K, H), self.covariance)

    def get_state(self):
        return self.state

# Initial state guess
initial_state = np.array([0, 0, 0, 0])  # [x, x_dot, y, y_dot]
initial_covariance = np.eye(4) * 1000  # Initial covariance matrix

# Process and measurement noise
process_noise = 0.1
measurement_noise = 1

# Initialize Kalman filter
kf = KalmanFilter2D(initial_state, initial_covariance, process_noise, measurement_noise)

# Simulate measurements
measurements = np.array([[1, 0], [2, 0.2], [3.1, 0.3], [4.2, 0.5], [5.1, 0.6]])

# Run Kalman filter
for measurement in measurements:
    kf.predict(dt=1)  # Assuming time step of 1 second
    kf.update(measurement)
    print("Estimated position:", kf.get_state()[:2])
