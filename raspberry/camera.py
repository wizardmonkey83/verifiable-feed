from config import PSQL_ADMIN_NAME, PSQL_ADMIN_PASS, PI_IP_ADDRESS, DB_NAME
import psycopg2
import cv2
import time

def build_cam_env():
    connection = psycopg2.connect(
        host=PI_IP_ADDRESS,
        database=DB_NAME,
        user=PSQL_ADMIN_NAME,
        password=PSQL_ADMIN_PASS
    )

    cursor = connection.cursor()

    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not cap.isOpened():
            return Exception("Error: could not open video.")

        time.sleep(2)

        motion_detected = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # color isn't useful for finding movement
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not gray_frame:
                return Exception("Error: Unable to convert frame to grayscale.")

            # second param defines kernel size which i guess is how blurred the image becomes (higher = more blur)
            blur_frame = cv2.blur(gray_frame, (5, 5))
            if not blur_frame:
                return Exception("Error: Unable to blur frame.")

            # detects prolonged environment to better detect anomolous movement
            mog2_func = cv2.createBackgroundSubtractorMOG2()
            mogged_frame = mog2_func.apply(blur_frame)

            count, binary_frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)
            if not binary_frame:
                return Exception("Error: Unable to run threshold on frame.")
            
            # contours docs: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
            contours, hierarchy = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            drawn_frame = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
            
            # area of countour docs: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
            changed_pixel_count = cv2.contourArea(contours)

            max_changed_count = 5000
            if changed_pixel_count > max_changed_count:
                motion_detected = True

            



    except Exception as e:
        return Exception(f"Error during build_cam_env: {e}")
    
    finally:
        if cap:
            cap.release()

        if cursor:
            cursor.close()

        if connection:
            connection.close()

if __name__ == "__main__":
    build_cam_env()