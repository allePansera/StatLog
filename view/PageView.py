from flask_classful import route, FlaskView
from flask import render_template, request


class PageView(FlaskView):

    def index(self):
        return render_template("landing_page.html")

    @route('/stat_log')
    def stat_log(self):
        return render_template("stat_log.html")
