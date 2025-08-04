# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
svm_loaded = joblib.load('catboost.pkl')
print("✅ Model loaded successfully (No Scaling Applied)!")

# Important columns used during training
important_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 
                  21, 22, 23, 24, 25, 26, 33, 34, 35, 36, 37, 38, 44, 46]

# Class Labels and Descriptions
class_descriptions = {
    0: "N (Normal) - 🟢 **Normal sinus rhythm** represents a healthy heartbeat, originating from the sinoatrial (SA) node. It ensures proper blood circulation and oxygen delivery throughout the body. A consistent and well-regulated rhythm, with a heart rate typically between 60-100 beats per minute, signifies good cardiovascular health. Disruptions in this rhythm may indicate underlying health issues such as heart disease, arrhythmias, or electrical conduction disorders. This is the ideal heart rhythm. 💓",
    
    1: "S (Supraventricular ectopic) - 🔴 **Supraventricular ectopic beats (SVEB)** are early heartbeats originating from the atria, rather than the ventricles. These beats can occur due to factors like stress, caffeine, alcohol, or certain medications. While they may cause palpitations or an irregular heartbeat, they are generally not dangerous unless they occur frequently or in conjunction with other heart conditions. SVEBs are often benign but should be monitored. ⚡️",
    
    2: "V (Ventricular ectopic) - ⚫ **Ventricular ectopic beats (VEB)** originate from the ventricles of the heart, which can cause a disruption in the regular rhythm. These abnormal beats may indicate underlying heart disease, especially if they occur frequently. VEBs can lead to more serious arrhythmias, such as ventricular tachycardia, and increase the risk of heart failure or sudden cardiac arrest. Monitoring is essential for individuals with frequent VEBs. 🏥💔",
    
    3: "F (Fusion of ventricular and normal) - 🟠 **Fusion beats** occur when a normal heartbeat coincides with a ventricular ectopic beat, resulting in a hybrid or 'fusion' signal. This is often seen in patients with conduction abnormalities or other heart diseases. Although fusion beats are generally considered less harmful than sustained ventricular arrhythmias, their occurrence can still signify an underlying electrical imbalance in the heart. ⚡️💔",
    
    4: "Q (Unknown beat) - ❓ **Unknown beats** refer to heartbeats that don't conform to any of the standard categories, making them difficult to classify. These could be due to artifacts, unusual rhythms, or undiagnosed conditions that require further examination. Unknown beats may indicate a need for deeper diagnostic testing or a reevaluation of the heart's electrical system. Investigating these beats can help in understanding complex arrhythmias. 🧐🧠"
}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/svm')
def svm():
    return render_template('svm.html')

@app.route('/knn')
def knn():
    return render_template('knn.html')

@app.route('/lr')
def lr():
    return render_template('lr.html')

@app.route('/gnb')
def gnb():
    return render_template('gnb.html')

@app.route('/dt')
def dt():
    return render_template('dt.html')

@app.route('/rf')
def rf():
    return render_template('rf.html')

@app.route('/ada')
def ada():
    return render_template('ada.html')

@app.route('/grad')
def grad():
    return render_template('grad.html')

@app.route('/xg')
def xg():
    return render_template('xg.html')

@app.route('/cat')
def cat():
    return render_template('cat.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read uploaded CSV file
        file = request.files['file']
        df = pd.read_csv(file, header=None, dtype=np.float64, encoding="utf-8")

        # Drop empty rows
        df = df.dropna()

        # Ensure correct feature count
        if df.shape[1] != 188:
            return render_template('index.html', error=f"Expected 188 features, but got {df.shape[1]}")

        # Select only the 30 features used during training
        X_new = df.iloc[:, important_cols].values.astype(np.float64)

        # Predict using SVM model (without scaling)
        predictions = svm_loaded.predict(X_new)
        class_predictions = [class_descriptions[pred] for pred in predictions]

        return render_template('index.html', predictions=class_predictions)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
