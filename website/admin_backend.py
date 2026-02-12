from flask import render_template, Blueprint, request, jsonify, session, flash, redirect, url_for, get_flashed_messages
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

base_dir = os.path.abspath(os.path.dirname(__file__))

admin_app = Blueprint("admin", __name__)

# Logic for admin page will be added later
@admin_app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if "admin" in session:
        return render_template("admin_page.html")
    
    if request.method == "POST":
        session["admin"] = "admin_name" # will get the admin name later
        return redirect(url_for("admin.admin_page"))
        
    return render_template("admin_login.html")

@admin_app.route("/admin_page")
def admin_page():
    if "admin" not in session:
        flash("Please log in to admin account first")
        return redirect(url_for("admin.admin_login"))
    return render_template("admin_page.html")

@admin_app.route("/model_testing")
def model_testing():
    if "admin" not in session:
        flash("Please log in to admin account first")
        return redirect(url_for("admin.admin_login"))
    return render_template("model_testing.html")

@admin_app.route("/nlp_model")
def nlp_model():
    if "admin" not in session:
        flash("Please log in to admin account first")
        return redirect(url_for("admin.admin_login"))
    return render_template("nlp_model.html")

@admin_app.route("/numerical_model")
def numerical_model():
    if "admin" not in session:
        flash("Please log in to admin account first")
        return redirect(url_for("admin.admin_login"))
    return render_template("numerical_model.html")

# file handling
@admin_app.route("/upload_files", methods=["POST"])
def upload_files():
    if "admin" not in session:
        flash("Please log in to admin account first")
        return redirect(url_for("admin.admin_login"))
    
    import pandas as pd
    
    start_time = time.time()
    
    uploaded_num = request.files.get("file_num")
    uploaded_nlp = request.files.get("file_nlp")
    
    uploaded_num = uploaded_num if uploaded_num is not None and uploaded_num.filename != "" else None
    uploaded_nlp = uploaded_nlp if uploaded_nlp is not None and uploaded_nlp.filename != "" else None
    
    confusion_num, confusion_nlp = "", ""
    
    accuracy_num, accuracy_nlp = 0, 0
    
    if uploaded_num is None and uploaded_nlp is None:
        error = "Please upload at least one file in one of the field"
        return jsonify({"error": error}), 400
    
    num_dict = {}
    nlp_dict = {}
    num_conf_matrix = {}
    nlp_conf_matrix = {} 
    num_model_tested = False
    nlp_model_tested = False
       
    if uploaded_num:
        df_num = pd.read_csv(uploaded_num, index_col=0)
        from shared_utils.num_inference import test_num_model
        
        confusion_num = test_num_model(df_num)
        
        for i in range(3):
            TP = confusion_num[i][i]
            FP = confusion_num[:, i].sum() - TP
            FN = confusion_num[i, :].sum() - TP
            
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            num_dict[f"class_{i - 1}"] = {"Precision": precision, "Recall": recall, "F1": f1}
            num_conf_matrix = confusion_num.tolist()
            
        accuracy_num = confusion_num.trace()  / confusion_num.sum()
        num_model_tested = True
        
    if uploaded_nlp:
        df_nlp = pd.read_csv(uploaded_nlp, index_col=0)
        from shared_utils.nlp_inference import test_nlp_model
        
        confusion_nlp = test_nlp_model(df_nlp)
        
        for i in range(3):
            TP = confusion_nlp[i][i]
            FP = confusion_nlp[:, i].sum() - TP
            FN = confusion_nlp[i, :].sum() - TP
            
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            nlp_dict[f"class_{i}"] = {"Precision": precision, "Recall": recall, "F1": f1}
            nlp_conf_matrix = confusion_nlp.tolist()
            
        accuracy_nlp = confusion_nlp.trace()  / confusion_nlp.sum()
        nlp_model_tested = True
        
    total_time = time.time() - start_time
    data_passed = {"num_dict": num_dict, "num_conf_matrix": num_conf_matrix, 
                   "nlp_dict": nlp_dict, "nlp_conf_matrix": nlp_conf_matrix,
                   "num_model_tested": num_model_tested, "nlp_model_tested": nlp_model_tested,
                   "accuracy_num": accuracy_num, "accuracy_nlp": accuracy_nlp, "total_time": total_time}
    
    return jsonify(data_passed), 200
            
        
        
        
