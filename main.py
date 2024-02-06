import numpy as np
import cv2
import dlib
import math
from skimage import metrics


def MSE(true, predicted):
    summation = 0
    n = len(true)
    for i in range(0, n):
        difference = true[i] - predicted[i]
        squared_difference = difference ** 2
        summation += squared_difference
    return summation / n


def SSIM_score(gray, current_points, predicted_points):
    g_pred = gray.copy()
    g_true = gray.copy()

    for p in current_points:
        cv2.circle(g_true, (int(p[0]), int(p[1])), 2, (255, 255, 255), -1)
    for p in predicted_points:
        cv2.circle(g_pred, (int(p[0]), int(p[1])), 2, (255, 255, 255), -1)

    return metrics.structural_similarity(g_pred, g_true, full=True)


def interEyeDistance(predict):
    leftEyeLeftCorner = (predict[36].x, predict[36].y)
    rightEyeRightCorner = (predict[45].x, predict[45].y)
    distance = cv2.norm(np.array(rightEyeRightCorner) - np.array(leftEyeLeftCorner))
    distance = int(distance)
    return distance


if __name__ == '__main__':
    # Capture Video Stream
    stream = cv2.VideoCapture(0)
    if not stream.isOpened():
        print("Camera connection / availability issue.")

    # Detect face
    face_detector = dlib.get_frontal_face_detector()

    # FL Predictor
    landmarks_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    # Initializing dummy arrays
    points, prev_points, curr_detections, prev_detections = [], [], [], []

    # Old frame for tracking
    _, old_frame = stream.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # dummy bool for first frame
    first_frame = True

    # Start destabilized
    stable = False

    # eye distance for calculation of window size of the LK algorithm
    eyeDistanceNotCalculated = True
    eyeDistance = 0

    # SSIMs = 0
    MSEs = [0, 0]
    detection_counter = 0

    while True:
        # frame capturing
        _, frame = stream.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # face detecting
        faces = face_detector(gray)
        if not faces:
            print("No faces found :(")
            if not first_frame:  # previous calculations
                prev_points = points
                prev_detections = curr_detections

                # Convert to numpy float array
                pointsArr = np.array(points, np.float32)
                prev_pointsArr = np.array(prev_points, np.float32)

                if eyeDistance > 100:
                    dotRadius = 3
                else:
                    dotRadius = 2

                sigma = eyeDistance * eyeDistance / 400
                s = 2 * int(eyeDistance / 4) + 1

                #  Set up optical flow params
                lk_params = dict(winSize=(s, s), maxLevel=2,
                                 criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03))

                pointsArr, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, prev_pointsArr, None, **lk_params)

                # Converting to float
                pointsArrFloat = np.array(pointsArr, np.float32)

                # Converting back to list
                points = pointsArrFloat.tolist()
                curr_detections = points

                # Showing stabilized in Green and destabilized in Red
                if stable:
                    for p in points:
                        cv2.circle(frame, (int(p[0]), int(p[1])), dotRadius, (0, 255, 0), -1)

                old_frame = frame
                old_gray = gray

                # Wait for ESC key to quit and SPACE key to stabilize/destiabilze the video
                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    stable = not stable
                if key == 27:  # ESC
                    stream.release()
                    # SSIMs /= detection_counter
                    # print("SSIM Score = ", round(SSIMs, 2))
                    MSEs /= detection_counter
                    print("MSE overall frames detected = ", MSEs)
                    break
        else:
            detection_counter += 1
            for i in range(0, len(faces)):
                newRect = dlib.rectangle(int(faces[i].left()), int(faces[i].top()), int(faces[i].right()),
                                         int(faces[i].bottom()))
                if first_frame:
                    [prev_points.append((p.x, p.y)) for p in landmarks_predictor(gray, newRect).parts()]
                    [prev_detections.append((p.x, p.y)) for p in landmarks_predictor(gray, newRect).parts()]
                else:
                    prev_points = points
                    prev_detections = curr_detections

                points = []
                curr_detections = []
                [points.append((p.x, p.y)) for p in landmarks_predictor(gray, newRect).parts()]
                [curr_detections.append((p.x, p.y)) for p in landmarks_predictor(gray, newRect).parts()]

                # Convert to numpy float array
                pointsArr = np.array(points, np.float32)
                prev_pointsArr = np.array(prev_points, np.float32)

                # If eye distance is not calculated before
                if eyeDistanceNotCalculated:
                    eyeDistance = interEyeDistance(landmarks_predictor(gray, newRect).parts())
                    eyeDistanceNotCalculated = False

                if eyeDistance > 100:
                    dotRadius = 3
                else:
                    dotRadius = 2

                sigma = eyeDistance * eyeDistance / 400
                s = 2 * int(eyeDistance / 4) + 1

                #  Set up optical flow params
                lk_params = dict(winSize=(s, s), maxLevel=5,
                                 criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03))

                pointsArr, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, prev_pointsArr, pointsArr, **lk_params)

                # Converting to float
                pointsArrFloat = np.array(pointsArr, np.float32)

                # Converting back to list
                points = pointsArrFloat.tolist()

                summation = 0
                # landmarks final are an average of the detected and the current
                for k in range(0, len(landmarks_predictor(gray, newRect).parts())):
                    d = cv2.norm(np.array(prev_detections[k]) - np.array(curr_detections[k]))
                    alpha = math.exp(-d * d / sigma)
                    points[k] = (1 - alpha) * np.array(curr_detections[k]) + alpha * np.array(points[k])

                    # for MSE calculation
                    summation += ((curr_detections[k] - points[k]) ** 2)

                # Calculate MSE of pixel shifts
                MSEs += summation / len(points)
                # MSEs += MSE(curr_detections, points)

                # Calculate SSIM
                # SSIMs += SSIM_score(gray, curr_detections, points)[0]

                # Showing stabilized in Green and destabilized in Red
                if stable:
                    for p in points:
                        cv2.circle(frame, (int(p[0]), int(p[1])), dotRadius, (0, 255, 0), -1)
                else:
                    for p in curr_detections:
                        cv2.circle(frame, (int(p[0]), int(p[1])), dotRadius, (0, 0, 255), -1)

                # set as false first frame was set
                first_frame = False

        # Show frame
        cv2.imshow("Press [SPACE] for stabilized landmarks, [ESC] to exit", cv2.flip(frame, 1))

        # assign old variable to current for next frame
        old_frame = frame
        old_gray = gray

        # Wait for ESC key to quit and SPACE key to stabilize/destiabilze the video
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            stable = not stable
        if key == 27:  # ESC
            stream.release()
            # SSIMs /= detection_counter
            # print("SSIM Score = ", round(SSIMs, 2))
            MSEs /= detection_counter
            print("MSE overall frames detected = ", MSEs)
            break
