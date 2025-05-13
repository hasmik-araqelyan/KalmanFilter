import cv2
import numpy as np

window_name = "Kalman Filter Mouse Tracker"
measured_points = []
kalman_points = []
predicted_points = []
frame = np.zeros((800, 800, 3), np.uint8)

state = np.zeros((4, 1), dtype=np.float32)  # [x, y, dx, dy]
P = np.eye(4, dtype=np.float32)  # Error covariance matrix
F = np.array([[1, 0, 1, 0],      # State transition matrix
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], dtype=np.float32)
H = np.array([[1, 0, 0, 0],      # Measurement matrix
              [0, 1, 0, 0]], dtype=np.float32)
Q = 1e-4 * np.eye(4, dtype=np.float32)  # Process noise covariance
R = 1e-1 * np.eye(2, dtype=np.float32)  # Measurement noise covariance

def predict():
    """Kalman prediction step"""
    global state, P
    
    # predict state
    state = np.dot(F, state) 

    # predict error covariance
    P = np.dot(np.dot(F, P), F.T) + Q 
    
    predicted_point = (state[0, 0], state[1, 0])
    predicted_points.append(predicted_point)
    
    return state.copy()

def correct(measurement):
    """Kalman correction step"""
    global state, P
    
    z = measurement.reshape(2, 1)
    
    #Kalman Gain
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    
    y = z - np.dot(H, state)

    # correct state
    state = state + np.dot(K, y)
    
    #correct error covariance
    P = np.dot(np.eye(4) - np.dot(K, H), P)

    corrected_point = (state[0, 0], state[1, 0])
    kalman_points.append(corrected_point)
    
    return state.copy()

def draw_trajectories():
    """Draw all trajectories on the frame"""
    # Draw measured trajectory
    for i in range(1, len(measured_points)):
        cv2.line(frame, measured_points[i-1], measured_points[i], (0, 100, 255), 2)
    
    # Draw corrected trajectory
    for i in range(1, len(kalman_points)):
        cv2.line(frame, (int(kalman_points[i-1][0]), int(kalman_points[i-1][1])),
                 (int(kalman_points[i][0]), int(kalman_points[i][1])), (0, 255, 0), 2)
    
    # Draw predicted trajectory
    for i in range(1, len(predicted_points)):
        cv2.line(frame, (int(predicted_points[i-1][0]), int(predicted_points[i-1][1])),
                 (int(predicted_points[i][0]), int(predicted_points[i][1])), (255, 0, 0), 2)

def draw_current_points():
    """Draw current points on the frame"""
    if len(measured_points) > 0:
        cv2.circle(frame, measured_points[-1], 5, (0, 0, 255), -1)
    if len(kalman_points) > 0:
        cv2.circle(frame, (int(kalman_points[-1][0]), int(kalman_points[-1][1])), 5, (0, 255, 0), -1)
    if len(predicted_points) > 0:
        cv2.circle(frame, (int(predicted_points[-1][0]), int(predicted_points[-1][1])), 5, (255, 0, 0), -1)

def draw_legend():
    """Draw legend on the frame"""
    cv2.putText(frame, "Measured (Red)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Corrected (Green)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Predicted (Blue)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

def draw_future_prediction(steps_ahead=5):
    """Draw future prediction on the frame"""
    if len(kalman_points) == 0:
        return
    
    future_points = []
    temp_state = state.copy()
    temp_P = P.copy()
    
    for _ in range(steps_ahead):
        # Predict state
        temp_state = np.dot(F, temp_state)

        # Predict covariance
        temp_P = np.dot(np.dot(F, temp_P), F.T) + Q
        future_points.append((temp_state[0, 0], temp_state[1, 0]))
    
    for i in range(len(future_points)-1):
        cv2.line(frame, 
                 (int(future_points[i][0]), int(future_points[i][1])),
                 (int(future_points[i+1][0]), int(future_points[i+1][1])),
                 (200, 0, 200), 2)
    
    cv2.putText(frame, f"Future Prediction ({steps_ahead} steps)", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

def mouse_callback(event, x, y, flags, param):
    global measured_points, frame
    
    if event == cv2.EVENT_MOUSEMOVE:
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        measured_points.append((x, y))
        
        predict()
        correct(measured)
        
        frame = np.zeros((800, 800, 3), np.uint8)
        
        draw_trajectories()
        draw_current_points()
        draw_legend()
        draw_future_prediction()

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

while True:
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('r'):  # Reset
        measured_points = []
        kalman_points = []
        predicted_points = []
        state = np.zeros((4, 1), dtype=np.float32)
        P = np.eye(4, dtype=np.float32)
        frame = np.zeros((800, 800, 3), np.uint8)

cv2.destroyAllWindows()