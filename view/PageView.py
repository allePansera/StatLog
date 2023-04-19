from flask_classful import route, FlaskView
from flask import render_template, request


class PageView(FlaskView):

    @route('/stat_log')
    def stat_log(self):
        return render_template("stat_log.html")
