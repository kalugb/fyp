from flask import render_template, Blueprint, request, jsonify, session, flash, redirect, url_for, get_flashed_messages
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

base_dir = os.path.abspath(os.path.dirname(__file__))

admin_app = Blueprint("admin", __name__)

@admin_app.before_request
def set_admin_mode_default():
    if "admin" not in session:
        session["admin_mode"] = False
        
# Logic for admin page will be added later
@admin_app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if "admin" in session:
        return render_template("admin_page.html")
    
    if request.method == "POST":
        session["admin"] = "admin_name" # will get the admin name later
        session["admin_mode"] = True
        return redirect(url_for("admin.admin_page"))
        
    return render_template("admin_login.html")

@admin_app.route("/admin_logout")
def admin_logout():
    session.pop("admin", None)
    session.pop("admin_mode", None)
    session["logout"] = True
    return redirect(url_for("home"))

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
    import numpy as np
    if "admin" not in session:
        flash("Please log in to admin account first")
        return redirect(url_for("admin.admin_login"))
    
    from shared_utils.load_utils import load_nlp_config
    
    data = load_nlp_config()
    
    model_name = data["model_name"]
    loss = data["loss_list"]
    loss_params = {"Loss Type": loss["type"], "Gamma": loss["gamma"], "Reduction Type": loss["reduction_type"], "Weights": loss["weights"]}
    params_list = data["hyperparameters_list"]
    params = {
        "Batch Size": params_list["batch_size"],
        "Learning Rate": params_list["lr"],
        "Weight Decay": params_list["weight_decay"],
        "Total Epochs": params_list["epochs"],
        "Threshold": f"{params_list["threshold_0"]:.4f}",
    }
    confusion = np.array(data["confusion_matrix"])
    metrics = {}
    
    for i in range(3):
        TP = confusion[i][i]
        FP = confusion[:, i].sum() - TP
        FN = confusion[i, :].sum() - TP        
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        metrics[f"class_{i - 1}"] = {
            "Precision": precision.item(),
            "Recall": recall.item(),
            "F1": f1.item()
        }
        
    accuracy = confusion.trace() / confusion.sum()
    metrics["accuracy"] = accuracy
    
    return render_template("nlp_model.html", model_name=model_name, loss_params=loss_params, 
                           params=params, metrics=metrics, accuracy=accuracy, confusion=confusion)

@admin_app.route("/numerical_model")
def numerical_model():
    import numpy as np
    if "admin" not in session:
        flash("Please log in to admin account first")
        return redirect(url_for("admin.admin_login"))
    
    from shared_utils.load_utils import load_num_config

    data = load_num_config()
    
    num_model_name = data["model_name"]
    params = data["params"]
    C = params["lr__C"]
    max_iter = params["lr__max_iter"]
    class_weight = params["class_weight"]
    labelled_class_weight = {
        "Short": class_weight["-1"],
        "Hold": class_weight["0"],
        "Long": class_weight["1"]
    }
    confusion = np.array(data["confusion_matrix"])
    backtest_result = data["backtesting_result"]
    metrics = {}
    
    for i in range(3):
        TP = confusion[i][i]
        FP = confusion[:, i].sum() - TP
        FN = confusion[i, :].sum() - TP        
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        metrics[f"class_{i - 1}"] = {
            "Precision": precision.item(),
            "Recall": recall.item(),
            "F1": f1.item()
        }
        
    accuracy = confusion.trace() / confusion.sum()
    metrics["accuracy"] = accuracy
    
    return render_template("numerical_model.html", model_name=num_model_name, 
                           C=C, max_iter=max_iter, class_weight=labelled_class_weight,
                           metrics=metrics, backtest_result=backtest_result, confusion=confusion)

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
            
        
        
        
