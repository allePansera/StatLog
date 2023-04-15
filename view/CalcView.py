from flask_classful import FlaskView, route
from flask import request, jsonify
from library.Appliance.TestCustomer import TestCustomer
import json


class CalcView(FlaskView):

    @route('/evaluate', methods=['POST'])
    def evaluate(self):
        form_data = dict(request.form)
        tc = TestCustomer()
        data = tc.normalize(form_data)
        tc.store_df(data)
        code = int(tc.predict(data.iloc[:, :])[0])
        return jsonify({"status": True, "code": code})
