from flask_classful import route, FlaskView
from flask import render_template, request


class PageView(FlaskView):

    @route('/home')
    def home(self):
        return render_template("home.html")
