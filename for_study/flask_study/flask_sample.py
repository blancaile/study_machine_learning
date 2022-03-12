from flask import Flask
import os

from flask.helpers import url_for

#app = Flask(__name__, static_folder='.',
#static_url_path="")

app = Flask(__name__)

@app.route("/hello")
def hello_world():
    return "Hello world"


@app.route("/")
def index():
    return url_for("show_user_profile",
    username="blancaile"
    )


#localhost:12345/hello/pythonでnameがpythonになる
@app.route("/user/<username>")
def show_user_profile(username):
    return "UserName: " + str(username)

@app.route("/post/<int:post_id>")
def show_post(post_id):
    return "Post" + str(post_id)

if __name__ == "__main__": 
    app.run(port=12345, debug=False)