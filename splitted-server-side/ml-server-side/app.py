from os import environ, makedirs

from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from sqlalchemy.exc import IntegrityError

from process_interview import process_interview

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret-key"
app.config["SQLALCHEMY_DATABASE_URI"] = environ.get("DB_URL")
db = SQLAlchemy(app)


def initialize():
    for dir in [
        "interview_outputs",
        "report_outputs",
        "videos",
        "processed_interviews",
    ]:
        try:
            makedirs(f"temp_files/{dir}")
        except FileExistsError:
            pass
    with app.app_context():
        try:
            db.session.execute(
                text(
                    """CREATE TABLE IF NOT EXISTS users
                        (id serial PRIMARY KEY, fullname varchar(50), email varchar(50), password varchar(150), 
                        location varchar(50), phone varchar(10), gender varchar(6))"""
                )
            )
        except IntegrityError:
            print("IntegrityError occured")

        gesture_names = [
            "angry",
            "bored",
            "disgust",
            "happy",
            "sad",
            "shy",
            "stressed",
            "surprised",
        ]
        try:
            db.session.execute(
                text(
                    f"""CREATE TABLE IF NOT EXISTS reports
                            (id varchar(32) PRIMARY KEY,
                            userid INTEGER REFERENCES users (id),
                            isfinished BOOL,
                            anomaliesNum integer,
                            title varchar(50),
                            date varchar(19),
                            {", ".join(list(map(lambda x : x + "percent FLOAT", gesture_names)))}
                            )"""
                )
            )
            db.session.commit()
        except IntegrityError:
            print("IntegrityError occured")


initialize()


def executeQuery(query):
    with app.app_context():
        db.session.execute(text(query))
        db.session.commit()


@app.route("/", methods=["POST"])
def upload():
    file_path = request.args.get("file_path")
    interviewId = request.args.get("interviewId")

    # processing video in the background
    process_interview(file_path, interviewId, executeQuery)

    return "", 204


if __name__ == "__main__":
    app.run(debug=True)
