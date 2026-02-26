from config import PSQL_ADMIN_NAME, PSQL_ADMIN_PASS, PI_IP_ADDRESS, DB_NAME
import psycopg2
import os
import cv2
import time, datetime
from picamera2 import Picamera2
from picamera2.outputs import CircularOutput
from picamera2.encoders import H264Encoder

RECORDING_DIR = "/home/admin/recordings"
os.makedirs(RECORDING_DIR, exist_ok=True)


# picamera2 docs: https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-2-picamera2-manual.pdf?disposition=inline
def build_cam_env():
    connection = psycopg2.connect(
        host=PI_IP_ADDRESS,
        database=DB_NAME,
        user=PSQL_ADMIN_NAME,
        password=PSQL_ADMIN_PASS
    )

    cursor = connection.cursor()

    try:

        picam2 = Picamera2()
        # creates two video streams, the lores one is passed through the while loop to run faster
        config = picam2.create_video_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480), "format": "RGB888"}, encode="lores")
        picam2.configure(config)
        picam2.start()

        encoder = H264Encoder(bitrate=2000000)
        ring_buffer = CircularOutput(buffersize=20000000)

        print("Starting pre-roll buffer...")
        picam2.start_recording(encoder, ring_buffer)

        time.sleep(2)


        # detects prolonged changes in background
        mog2_func = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        max_changed_count = 5000

        while True:
            frame = picam2.capture_array("lores")

            # color isn't useful for finding movement
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # second param defines kernel size which i guess is how blurred the image becomes (higher = more blur)
            blur_frame = cv2.blur(gray_frame, (5, 5))

            # background subtractor
            mogged_frame = mog2_func.apply(blur_frame)

            _, binary_frame = cv2.threshold(mogged_frame, 127, 255, cv2.THRESH_BINARY)
            
            # contours docs: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
            contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            drawn_frame = cv2.drawContours(frame, contours, -1, (0,255,0), 3)

            motion_detected = False
            
            for cont in contours:
                # area of countour docs: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
                changed_pixel_count = cv2.contourArea(cont)
                if changed_pixel_count > max_changed_count:
                    motion_detected = True
                    break

            if motion_detected:
                print("Motion detected, saving event...")

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                preroll_file = os.path.join(RECORDING_DIR, f"preroll_{timestamp}.h264")
                active_file = os.path.join(RECORDING_DIR, f"active_{timestamp}.h264")

                with open(preroll_file, "wb") as f:
                    for buf in ring_buffer.read():
                        f.write(buf)
                ring_buffer.clear()

                picam2.stop_recording()
                picam2.start_recording(encoder, active_file)

                while motion_detected:
                    frame = picam2.capture_array("lores")
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blur_frame = cv2.blur(gray_frame, (5, 5))

                    # learningRate allows for the frame to be compared, but not used for altering the baseline
                    mogged_frame = mog2_func.apply(blur_frame, learningRate=0)

                    _, binary_frame = cv2.threshold(mogged_frame, 127, 255, cv2.THRESH_BINARY)

                    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    drawn_frame = cv2.drawContours(frame, contours, -1, (0,255,0), 3)

                    changed_motion = True
                    for cont in contours:
                        # area of countour docs: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
                        changed_pixel_count = cv2.contourArea(cont)
                        if changed_pixel_count > max_changed_count:
                            last_motion_time = time.time()
                            if time.time() - last_motion_time > 5.0:
                                changed_motion = False
                                break

                    if changed_motion:
                        motion_detected = False

                picam2.stop_recording()
                picam2.start_recording(encoder, ring_buffer)

                print(f"Event saved: {active_file}.")






            



    except Exception as e:
        print(f"Error during build_cam_env: {e}")
    
    finally:
        picam2.stop()
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == "__main__":
    build_cam_env()