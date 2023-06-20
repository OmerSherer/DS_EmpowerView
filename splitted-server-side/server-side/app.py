from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os import environ, makedirs

from requests import post
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    redirect,
    send_file,
    url_for,
    session,
)
from werkzeug.security import generate_password_hash, check_password_hash
import uuid

from numpy import array as np_array
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from sqlalchemy.exc import IntegrityError

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret-key"
app.config["SQLALCHEMY_DATABASE_URI"] = environ.get("DB_URL")
db = SQLAlchemy(app)


def executeQuery(query):
    with app.app_context():
        try:
            db.session.execute(text(query))
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            print(f"IntegrityError occured in query: \n{query}\n")


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
    executeQuery(
        """CREATE TABLE IF NOT EXISTS users
                (id serial PRIMARY KEY, fullname varchar(50), email varchar(50), password varchar(150), 
                location varchar(50), phone varchar(10), gender varchar(6))"""
    )

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
    executeQuery(
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


initialize()


def fetchQuery(query):
    with app.app_context():
        return db.session.execute(text(query))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/signup", methods=["POST", "GET"])
def signup():
    if request.method == "POST":
        fullname = request.form["fullname"]
        email = request.form["email"]
        password = generate_password_hash(
            request.form["password"], method="pbkdf2:sha256"
        )
        location = request.form["location"]
        phone = request.form["phone"]
        gender = request.form["gender"]

        executeQuery(
            f"INSERT INTO users (fullname, email, password, location, phone, gender) VALUES ('{fullname}', '{email}', '{password}', '{location}', '{phone}', '{gender}')",
        )

        return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/interview")
def interview():
    if "user_id" not in session:
        return redirect(url_for("login"))

    result = fetchQuery(f"SELECT * FROM users WHERE id = {session['user_id']}")
    user = result.fetchone()

    if user is None:
        return redirect(url_for("login"))

    return render_template("interview.html", fullname=user[1])


executor = ThreadPoolExecutor(max_workers=1)


@app.route("/interview/upload", methods=["POST"])
def upload():
    video = request.files["file-upload"]

    interviewId = uuid.uuid4().hex

    # saving the video
    # TODO: remove the files on finish
    file_folder = "temp_files/videos/"
    file_name = f"{interviewId}.mp4"
    file_path = file_folder + file_name

    video.save(file_path)

    # processing video in the background
    executeQuery(
        f"""INSERT INTO Reports (id, userid, isfinished, title, date) VALUES (
            '{interviewId}',
            {session["user_id"]},
            FALSE,
            '{request.form["name"]}',
            '{datetime.now().replace(microsecond=0).isoformat()}'
        )""",
    )

    url = (
        environ.get("ML_URL")
        if environ.get("ML_URL") is not None
        else "http://empowerview_server_ml:3000"
    )

    params = {"file_path": file_path, "interviewId": interviewId}

    post(url, params=params)  # type: ignore

    return redirect(url_for("report"))


# @app.route('/interview', methods=['POST'])
# def interview():
#     video = request.files['file-upload']
#     video.save('/home/cs206/Downloads/video.mp4')
#     return 'Video uploaded and saved successfully!'


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        result = fetchQuery(f"SELECT * FROM users WHERE email = '{email}'")
        user = result.fetchone()

        if user and check_password_hash(user[3], password):
            session["user_id"] = user[0]
            return redirect(url_for("welcome"))

        return render_template(
            "login.html", error_message="Invalid username or password"
        )
    else:
        return render_template("login.html")


@app.route("/welcome")
def welcome():
    if "user_id" not in session:
        return redirect(url_for("login"))

    result = fetchQuery(f"SELECT * FROM users WHERE id = {session['user_id']}")
    user = result.fetchone()

    if user is None:
        return redirect(url_for("login"))

    return render_template("welcome.html", fullname=user[1])


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


def is_authenticated():
    if "user_id" not in session:
        return False

    result = fetchQuery(f"SELECT * FROM users WHERE id = {session['user_id']}")
    user = result.fetchone()

    if user is None:
        return False
    return user


@app.route("/api/reports")
def apiReports():
    if not is_authenticated():
        return "", 403

    result = fetchQuery(f"SELECT * FROM Reports WHERE userid = {session['user_id']}")
    reports = result.fetchall()
    keyedReports = []
    for report in reports:
        keyedReports.append({})
        for key, value in zip(
            [
                "id",
                "userid",
                "isfinished",
                "anomaliesNum",
                "title",
                "date",
                "angrypercent",
                "boredpercent",
                "disgustpercent",
                "happypercent",
                "sadpercent",
                "shypercent",
                "stressedpercent",
                "surprisedpercent",
            ],
            report,
        ):
            if "userid" == key:
                continue
            if "isfinished" == key:
                keyedReports[-1][key] = bool(value)
            elif key[-7:] == "percent" and value is not None:
                keyedReports[-1][key] = round(value * 100, ndigits=1)
            else:
                keyedReports[-1][key] = value
    return jsonify(keyedReports)


@app.route("/api/reports/anomalypercent/<id>")
def apiAnomalyPercent(id):
    if not is_authenticated():
        return "", 403

    result = fetchQuery(f"SELECT * FROM AnomliesByTime_{id}")
    anomalyPercentges = result.fetchall()
    anomalyPercentgesT = np_array(anomalyPercentges).T
    return jsonify(
        {
            "labels": list(anomalyPercentgesT[0].round(decimals=2)),
            "values": list(anomalyPercentgesT[1].round(decimals=2)),
        }
    )


@app.route("/api/reports/confidence/<id>")
def apiConfidence(id):
    if not is_authenticated():
        return "", 403

    result = fetchQuery(
        f"SELECT timestamp, angry, bored, disgust, happy, sad, shy, stressed, surprised FROM ConfidencesByTime_{id}"
    )
    confidence = result.fetchall()
    confidenceT = np_array(confidence).T
    returnedConfidence = {}
    emojis = {
        "angry": "😠",
        "bored": "🥱",
        "disgust": "🤢",
        "happy": "😀",
        "sad": "😥",
        "shy": "🥺",
        "stressed": "💦",
        "surprised": "😨",
    }
    for confidence, label in zip(
        confidenceT,
        [
            "timestamp",
            "angry",
            "bored",
            "disgust",
            "happy",
            "sad",
            "shy",
            "stressed",
            "surprised",
        ],
    ):
        if label == "timestamp":
            returnedConfidence[label] = list((confidence).round(decimals=2))
            continue
        returnedConfidence[emojis[label] + label] = list(
            (confidence * 100).round(decimals=1)
        )
    return jsonify(returnedConfidence)


@app.route("/api/reports/general/<id>")
def apiGeneral(id):
    if not is_authenticated():
        return "", 403

    result = fetchQuery(
        f"SELECT angrypercent, boredpercent, disgustpercent, happypercent, sadpercent, shypercent, stressedpercent, surprisedpercent FROM Reports WHERE id = '{id}'"
    )
    general = result.fetchone()
    generalData = []

    for data in general:
        generalData.append(round(data * 100, ndigits=1))
    return jsonify(
        {
            "labels": [
                "😠angry",
                "🥱bored",
                "🤢disgust",
                "😀happy",
                "😥sad",
                "🥺shy",
                "💦stressed",
                "😨surprised",
            ],
            "data": generalData,
        }
    )


@app.route("/api/reports/video/<id>")
def apiVideo(id):
    if not is_authenticated():
        return "", 403

    # TODO: protect from file injection
    return send_file(
        f"temp_files/processed_interviews/{id}.mp4",
        as_attachment=True,
        download_name=f"{id}.mp4",
    )


@app.route("/reports")
def report():
    user = is_authenticated()
    if not user:
        return redirect(url_for("login"))

    return render_template("report.html", fullname=user[1])


@app.route("/reports/<reportid>")
def detailedReport(reportid):
    user = is_authenticated()
    if not user:
        return redirect(url_for("login"))

    return render_template("detailedReport.html", fullname=user[1], reportid=reportid)


if __name__ == "__main__":
    app.run(debug=True)
