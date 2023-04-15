from flask import Flask
from view.PageView import PageView
from view.CalcView import CalcView

app = Flask(__name__)
PageView.register(app)
CalcView.register(app)
