from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import uuid

from process_interview import process_interview

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
    # TODO: add column that will identify the report for the user (by name, date, etc.)
    c.execute(
        f"""CREATE TABLE IF NOT EXISTS Reports
                    (id TEXT PRIMARY KEY,
                    userid INTEGER,
                    isfinished INTEGER,
                    anomaliesNum INTEGER,
                    {", ".join(list(map(lambda x : x + "precent REAL", gesture_names)))},
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
    file_folder = "temp_files/videos/"
    file_name = f"{interviewId}.mp4"
    file_path = file_folder + file_name

    video.save(file_path)

    # processing video in the background
    process_interview(file_path, interviewId)

    return "Video uploaded and saved successfully!"


# @app.route('/interview', methods=['POST'])
# def interview():
#     video = request.files['file-upload']
#     video.save('/home/cs206/Downloads/video.mp4')
#     return 'Video uploaded and saved successfully!'


@app.route("/report")
def report():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM User WHERE id = ?", (session["user_id"],))
    user = c.fetchone()
    conn.close()

    if user is None:
        return redirect(url_for("report"))

    return render_template("report.html", fullname=user[1])


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
from process_interview import func

func(app)
########

if __name__ == "__main__":
    app.run(debug=True)
