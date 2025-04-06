import pandas as pd

# Load and preprocess data
df=pd.read_csv(r"D:\6th_sem_project\ASD_dataset.csv")
df = df.rename(columns={"Age_Mons":"Age Months",
                        "Family_mem_with_ASD":"Family Member with ASD",
                        "Class/ASD Traits ": "ASD Traits"})
df.drop("Case_No", axis=1, inplace=True)
df["Age"] = df["Age Months"] / 12
df.drop("Age Months", axis=1, inplace=True)
df.drop(['Ethnicity','Who completed the test'], axis=1, inplace=True)

# Encode categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Jaundice"] = le.fit_transform(df["Jaundice"])
df["Family Member with ASD"] = le.fit_transform(df["Family Member with ASD"])
df["ASD Traits"] = le.fit_transform(df["ASD Traits"])

# Split data
X = df.drop(columns=["ASD Traits"])
y = df["ASD Traits"]
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = MinMaxScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
order=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
       'Qchat-10-Score', 'Age', 'Sex', 'Jaundice', 'Family Member with ASD']
X=X[order]
print(X.columns)

# Train and evaluate model
'''import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(32, activation="relu", input_dim=X_train_scaled.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1), metrics=["accuracy"])

callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
model.fit(X_train_scaled, y_train, batch_size=10, epochs=70, validation_split=0.2, callbacks=callback)

model.save("ann.keras")'''

from flask import Flask, render_template, request, flash, redirect, url_for
import tensorflow as tf
import numpy as np
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("ann.keras")
background_data=X_test_scaled
feature_name=df.columns


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/approach')
def approach():
    return render_template('approach.html')


@app.route('/solution', methods=['GET', 'POST'])
def solution():
    if request.method == 'POST':
        try:
        # Collect form data
            a1 = int(request.form['a1'])
            a2 = int(request.form['a2'])
            a3 = int(request.form['a3'])
            a4 = int(request.form['a4'])
            a5 = int(request.form['a5'])
            a6 = int(request.form['a6'])
            a7 = int(request.form['a7'])
            a8 = int(request.form['a8'])
            a9 = int(request.form['a9'])
            a10 = int(request.form['a10'])
            q_chat = int(request.form['qchat'])
            age = int(request.form['age'])

            sex = int(request.form['sex'])
            jaundice = int(request.form['jaundice'])
            family_asd = int(request.form['family_asd'])

        # Prepare input data for the model
            input_data = [
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, q_chat,
                age, sex, jaundice, family_asd
            ]

        # Convert input data to NumPy array and reshape it
            #input_array = np.array(input_data).reshape(1, -1)
            #input_array=sc.transform(input_array)
            input_array = pd.DataFrame([input_data], columns=X.columns)
            #input_array = sc.transform(input_df)

        # Get the prediction probability
            prediction_proba = model.predict(input_array)[0][0]

            probability_text = f"Probability: {prediction_proba * 100:.2f}%"

            explainer = shap.Explainer(model, background_data)
            shap_values = explainer(input_array)


            buf = BytesIO()
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()

            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()


            if prediction_proba > 0.5:

                treatments = [
                "Behavioral Therapy",
                "Speech Therapy",
                "Occupational Therapy",
                "Medication",
                "Social Skills Training"
            ]
                recommendation_message = f"High Risk of ASD. {probability_text}"
            else:
                treatments = []
                recommendation_message = (f"Low Risk of ASD. {probability_text} <br>"
                                          f"No treatments needed, Still consult a professional expert for better understanding.")

            return render_template('recommendation.html', recommendation_message=recommendation_message,
                               treatments=treatments, plot_data=plot_data)
        except Exception as e:
            flash('Error processing your request. Please try again.')
            return redirect(url_for('solution'))
    return render_template('solution.html')


@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html', recommendation_message="No prediction yet.", treatments=[], plot_data=None)

'''explainer = shap.Explainer(model, background_data)
sample_input=X_test_scaled[:20]
shap_values = explainer(sample_input)
plt.figure()
shap.waterfall_plot(shap_values[0])
buf = BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
buf.close()
print("Length of plot_data:", len(plot_data))
print("First 200 characters:", plot_data[:200])'''



if __name__ == '__main__':
    app.run(debug=True)

