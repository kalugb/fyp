from flask import render_template, Blueprint, request, jsonify
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

base_dir = os.path.abspath(os.path.dirname(__file__))

admin_app = Blueprint("admin", __name__)

@admin_app.route("/admin_login")
def admin_login():
    return render_template("admin_login.html")

@admin_app.route("/admin_page")
def admin_page():
    return render_template("admin_page.html")

@admin_app.route("/model_testing")
def model_testing():
    return render_template("model_testing.html")

@admin_app.route("/nlp_model")
def nlp_model():
    return render_template("nlp_model.html")

@admin_app.route("/numerical_model")
def numerical_model():
    return render_template("numerical_model.html")

# file handling
@admin_app.route("/upload_files", methods=["POST"])
def upload_files():
    import pandas as pd
    
    uploaded_num = request.files.get("file_num")
    uploaded_nlp = request.files.get("file_nlp")
    
    uploaded_num = uploaded_num if uploaded_num is not None and uploaded_num.filename != "" else None
    uploaded_nlp = uploaded_nlp if uploaded_nlp is not None and uploaded_nlp.filename != "" else None
    
    if uploaded_num is None and uploaded_nlp is None:
        error = "Please upload at least one file in one of the field"
        return jsonify({"error": error}), 400
    
    if uploaded_num:
        pass
    
    if uploaded_nlp:
        pass
    
    return jsonify({"num_precision": "number", "num_recall": "number2", "num_f1": "number3",
                    "nlp_precision": "number4", "nlp_recall": "number5", "nlp_f1": "number6"})
            
        
        
        
