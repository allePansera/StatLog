from flask import Flask
from view.PageView import PageView
from view.CalcView import CalcView
from view.GitView import GitView


app = Flask(__name__)
PageView.register(app)
CalcView.register(app)
GitView.register(app)

