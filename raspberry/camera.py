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

        dequeue = []
        while dequeue:
            ret, frame = cap.read()
            if not ret:
                break

            if len(dequeue) >= 150:
                dequeue.pop()
            
            dequeue.insert(0, frame)

            

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