import numpy as np
import cv2
import dlib
import math


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
    points, pointsPrev, pointsDetectedCur, pointsDetectedPrev = [], [], [], []

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

    while True:
        # frame capturing
        _, frame = stream.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # face detecting
        faces = face_detector(gray)
        if not faces:
            print("No faces found :( ")
        else:
            for i in range(0, len(faces)):
                newRect = dlib.rectangle(int(faces[i].left()), int(faces[i].top()), int(faces[i].right()),
                                         int(faces[i].bottom()))
                if first_frame:
                    [pointsPrev.append((p.x, p.y)) for p in landmarks_predictor(gray, newRect).parts()]
                    [pointsDetectedPrev.append((p.x, p.y)) for p in landmarks_predictor(gray, newRect).parts()]
                else:
                    pointsPrev = []
                    pointsDetectedPrev = []
                    pointsPrev = points
                    pointsDetectedPrev = pointsDetectedCur

                points = []
                pointsDetectedCur = []
                [points.append((p.x, p.y)) for p in landmarks_predictor(gray, newRect).parts()]
                [pointsDetectedCur.append((p.x, p.y)) for p in landmarks_predictor(gray, newRect).parts()]

                # Convert to numpy float array
                pointsArr = np.array(points, np.float32)
                pointsPrevArr = np.array(pointsPrev, np.float32)

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

                points = []
                pointsDetectedCur = []
                [points.append((p.x, p.y)) for p in landmarks_predictor(frame, newRect).parts()]
                [pointsDetectedCur.append((p.x, p.y)) for p in landmarks_predictor(frame, newRect).parts()]

                pointsArr, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, pointsPrevArr, pointsArr, **lk_params)

                # Converting to float
                pointsArrFloat = np.array(pointsArr, np.float32)

                # Converting back to list
                points = pointsArrFloat.tolist()

                # landmarks final are an average of the detected and the current
                for k in range(0, len(landmarks_predictor(gray, newRect).parts())):
                    d = cv2.norm(np.array(pointsDetectedPrev[k]) - np.array(pointsDetectedCur[k]))
                    alpha = math.exp(-d * d / sigma)
                    points[k] = (1 - alpha) * np.array(pointsDetectedCur[k]) + alpha * np.array(points[k])

                # Showing stabilized in Green and destabilized in Red
                if stable:
                    for p in points:
                        cv2.circle(frame, (int(p[0]), int(p[1])), dotRadius, (0, 255, 0), -1)
                else:
                    for p in pointsDetectedCur:
                        cv2.circle(frame, (int(p[0]), int(p[1])), dotRadius, (0, 0, 255), -1)

                # set as false first frame was set
                first_frame = False

        # Show frame
        cv2.imshow("Video Stream", cv2.flip(frame, 1))

        # assign old variable to current for next frame
        old_frame = frame
        old_gray = gray

        # Wait for ESC key to quit and SPACE key to stabilize/destiabilze the video
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            stable = not stable
        if key == 27:  # ESC
            stream.release()
            break
