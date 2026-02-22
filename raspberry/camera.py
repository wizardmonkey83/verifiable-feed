from config import PSQL_ADMIN_NAME, PSQL_ADMIN_PASS, PI_IP_ADDRESS, DB_NAME
import psycopg2
import cv2
from time import sleep

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

        if not cap.isOpened():
            return Exception("Error: could not open video.")
            exit()

        cursor.execute("INSERT INTO recordings ")

    except Exception as e:
        return Exception(f"Error during build_cam_env: {e}")

    cursor.execute()

if __name__ == "__main__":
    build_cam_env()