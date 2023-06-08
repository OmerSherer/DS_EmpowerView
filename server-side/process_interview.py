import sqlite3
import threading

from classify import process_video_to_csv  # gesture classifyer
from report import make_report  # report maker
from anomalyDetection import analyzeVideo  # annomaly detection


def func(app):
    @app.route("/hello")
    def hello():
        return "hello"


def process_interview(file_path, interviewId, uploaderId):
    def process_interview_thread(file_path, interviewId, uploaderId):
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        # TODO: add actual userId
        c.execute(
            "INSERT INTO Reports (id, userid, isfinished) VALUES (?, ?, ?)",
            (interviewId, uploaderId, False),
        )
        conn.commit()
        conn.close()

        fps = process_video_to_csv(
            input_file=file_path,
            model_path="models/my_model6.h5",
            output_file_coords=f"temp_files/interview_outputs/{interviewId}-coords.csv",
            output_file_confidence=f"temp_files/interview_outputs/{interviewId}-confidence.csv",
            show_cam=False,
        )

        # make_report(
        #     f"temp_files/interview_outputs/{interviewId}-confidence.csv")

        (
            numOfAnomlies,
            _,
            xAnomlies,
            yAnomlies,
            y_predictions,
            expectedPoints,
            actualPoints,
        ) = analyzeVideo(f"temp_files/interview_outputs/{interviewId}-coords.csv", fps)
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

    interview_thread = threading.Thread(
        target=process_interview_thread, args=(
            file_path, interviewId, uploaderId)
    )
    interview_thread.start()
