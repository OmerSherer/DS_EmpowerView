from datetime import datetime
from queue import Queue
import cv2
import sqlite3
import threading

from classify import process_video_to_csv  # gesture classifyer
from report import make_report, insert_confidences_to_tables  # report maker
from anomalyDetection import analyzeVideo, writeAnomaliesOnFrame  # annomaly detection


class SingletonQueue:
    _instance = None
    _lock = threading.Lock()

    @staticmethod
    def get_instance():
        if not SingletonQueue._instance:
            with SingletonQueue._lock:
                if not SingletonQueue._instance:
                    SingletonQueue._instance = SingletonQueue()
        return SingletonQueue._instance

    def __init__(self):
        self.task_queue = Queue()
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def add_task(self, task_func):
        self.task_queue.put(task_func)

    def _process_queue(self):
        while True:
            try:
                task_func = self.task_queue.get()
                task_func()
                self.task_queue.task_done()
            except KeyboardInterrupt:
                break


def func(app):
    @app.route("/hello")
    def hello():
        return "hello"


def process_interview(file_path, interviewId, uploaderId, name):
    def process_interview_thread(file_path, interviewId):
        # processing the input video into a co-ordinates csv file and a confidence csv file (classifier output)
        (fps, df_coords, df_confidence) = process_video_to_csv(
            input_file=file_path,
            model_path="models/my_model6.h5",
            output_file_coords=f"temp_files/interview_outputs/{interviewId}-coords.csv",
            output_file_confidence=f"temp_files/interview_outputs/{interviewId}-confidence.csv",
            show_cam=False,
        )

        # make_report(
        #     f"temp_files/interview_outputs/{interviewId}-confidence.csv")
        insert_confidences_to_tables(
            df_confidence=df_confidence, interviewId=interviewId
        )

        # annomaly detection
        (
            numOfAnomlies,
            _,
            xAnomlies,
            yAnomlies,
            y_predictions,
            expectedPoints,
            actualPoints,
        ) = analyzeVideo(f"temp_files/interview_outputs/{interviewId}-coords.csv", fps)

        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        border_size = 10
        out = cv2.VideoWriter(
            f"./videos/{interviewId}.mp4",
            fourcc,
            float(fps),
            (int(cap.get(3)) + border_size * 2, int(cap.get(4)) + border_size * 2),
        )

        index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_confidence = df_confidence.iloc[index]

            frame = cv2.putText(
                frame,
                current_confidence[1]
                + ": {:.1f}%".format(max(current_confidence[2:]) * 100),
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            frame, isAnomaly = writeAnomaliesOnFrame(
                cap,
                frame,
                index,
                border_size,
                y_predictions,
                expectedPoints,
                actualPoints,
            )

            if isAnomaly:
                for _ in range(int(fps)):
                    out.write(frame)
            else:
                out.write(frame)

            index += 1

        cap.release()
        out.release()

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute(
            """
            UPDATE Reports
            SET isfinished = ?,
                anomaliesNum = ?
            WHERE
                id = ?
            """,
            (True, int(numOfAnomlies), interviewId),
        )
        c.execute(
            f"""CREATE TABLE IF NOT EXISTS AnomliesByTime_{interviewId}
                 (time REAL, chance REAL)"""
        )
        for time, chance in zip(xAnomlies, yAnomlies):
            c.execute(
                f"""INSERT INTO anomliesByTime_{interviewId} (time, chance)
                VALUES (?, ?)""",
                (float(time), float(chance)),
            )
        conn.commit()
        conn.close()

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO Reports (id, userid, isfinished, title, date) VALUES (?, ?, ?, ?, ?)",
        (
            interviewId,
            uploaderId,
            False,
            name,
            datetime.now().replace(microsecond=0).isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    SingletonQueue.get_instance().add_task(
        lambda: process_interview_thread(file_path, interviewId)
    )
