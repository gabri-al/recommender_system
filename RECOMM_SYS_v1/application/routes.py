from application import app

@app.route("/")
def index():
    return "<h1>Recommender System is coming!</h1>"
