from process_interview import func
from flask import Flask, jsonify, render_template, request, redirect, send_file, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import uuid

from process_interview import process_interview
from numpy import array as np_array
from numpy import round as to_np_round_array

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret-key"


def create_tables():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS User
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, fullname TEXT, email TEXT, password TEXT, 
                  location TEXT, phone TEXT, gender TEXT)"""
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
    c.execute(
        f"""CREATE TABLE IF NOT EXISTS Reports
                    (id TEXT PRIMARY KEY,
                    userid INTEGER,
                    isfinished INTEGER,
                    anomaliesNum INTEGER,
                    title TEXT,
                    date TEXT,
                    {", ".join(list(map(lambda x : x + "percent REAL", gesture_names)))},
                    FOREIGN KEY(userid) REFERENCES User(id)
                    )"""
    )
    conn.commit()
    conn.close()


create_tables()


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

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute(
            "INSERT INTO User (fullname, email, password, location, phone, gender) VALUES (?, ?, ?, ?, ?, ?)",
            (fullname, email, password, location, phone, gender),
        )
        conn.commit()
        conn.close()

        return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/interview")
def interview():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM User WHERE id = ?", (session["user_id"],))
    user = c.fetchone()
    conn.close()

    if user is None:
        return redirect(url_for("login"))

    return render_template("interview.html", fullname=user[1])


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
    process_interview(file_path, interviewId,
                      session["user_id"], request.form['name'])

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

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("SELECT * FROM User WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()

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

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM User WHERE id = ?", (session["user_id"],))
    user = c.fetchone()
    conn.close()

    if user is None:
        return redirect(url_for("login"))

    return render_template("welcome.html", fullname=user[1])


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


# example

func(app)
########


def is_authenticated():
    if "user_id" not in session:
        return False

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM User WHERE id = ?", (session["user_id"],))
    user = c.fetchone()
    conn.close()

    if user is None:
        return False
    return user


@app.route("/api/reports")
def apiReports():
    if not is_authenticated():
        return "", 403

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM Reports WHERE userid = ?", (session["user_id"],))
    reports = c.fetchall()
    conn.close()
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

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(f"SELECT * FROM AnomliesByTime_{id}")
    anomalyPercentges = c.fetchall()
    conn.close()
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

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        f"SELECT timestamp, angry, bored, disgust, happy, sad, shy, stressed, surprised FROM ConfidencesByTime_{id}")
    confidence = c.fetchall()
    conn.close()
    confidenceT = np_array(confidence).T
    returnedConfidence = {}
    emojis = {"angry": "😠", "bored": "🥱", "disgust": "🤢", "happy": "😀",
              "sad": "😥", "shy": "🥺", "stressed": "💦", "surprised": "😨"}
    for confidence, label in zip(confidenceT, ["timestamp", "angry", "bored", "disgust", "happy", "sad", "shy", "stressed", "surprised"]):
        if label == "timestamp":
            returnedConfidence[label] = list((confidence).round(decimals=2))
            continue
        returnedConfidence[emojis[label] +
                           label] = list((confidence * 100).round(decimals=1))
    return jsonify(returnedConfidence)


@app.route("/api/reports/general/<id>")
def apiGeneral(id):
    if not is_authenticated():
        return "", 403

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT angrypercent, boredpercent, disgustpercent, happypercent, sadpercent, shypercent, stressedpercent, surprisedpercent FROM Reports WHERE id = ?", (id,))
    general = c.fetchone()
    conn.close()
    generalData = []

    for data in general:
        generalData.append(round(data * 100, ndigits=1))
    return jsonify({"labels": ["😠angry", "🥱bored", "🤢disgust", "😀happy", "😥sad", "🥺shy", "💦stressed", "😨surprised"], "data": generalData})


@app.route("/api/reports/video/<id>")
def apiVideo(id):
    if not is_authenticated():
        return "", 403

    # TODO: protect from file injection
    return send_file(f"./videos/{id}.mp4", as_attachment=True, download_name=f"{id}.mp4")


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
