from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("my_model7.keras")


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
        a10 = float(request.form['a10'])

        age = int(request.form['age'])
        # social_scale=int(request.form['social_scale'])
        q_chat = int(request.form['qchat'])
        '''speech_delay = int(request.form['speech_delay'])
        learning_disorder = int(request.form['learning_disorder'])
        genetic_disorders = int(request.form['genetic_disorders'])
        depression = int(request.form['depression'])
        developmental_delay = int(request.form['developmental_delay'])
        social_issues = int(request.form['social_issues'])
        autism_rating = float(request.form['autism_rating'])

        anxiety = int(request.form['anxiety'])'''

        sex = int(request.form['sex'])
        jaundice = int(request.form['jaundice'])
        family_asd = int(request.form['family_asd'])

        # Prepare input data for the model
        input_data = [
            a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, q_chat,
            age, sex, jaundice, family_asd
        ]

        # Convert input data to NumPy array and reshape it
        input_data = np.array(input_data).reshape(1, -1)

        # Get the prediction probability
        prediction_proba = model.predict(input_data)[0][0]

        probability_text = f"Probability: {prediction_proba * 100:.2f}%"

        # Render the recommendation page with the prediction result

        # Render the recommendation page with the prediction result
        if prediction_proba > 0.5:
            treatments = [
                "Behavioral Therapy",
                "Speech Therapy",
                "Occupational Therapy",
                "Medication",
                "Social Skills Training"
            ]
            recommendation_message = f"High Risk of ASD. Probability: {probability_text}"
        else:
            treatments = []
            recommendation_message = f"Low Risk of ASD. Probability: {probability_text}. No treatments needed, Still consult a professional expert for better understanding."

        return render_template('recommendation.html', recommendation_message=recommendation_message,
                               treatments=treatments)

    return render_template('solution.html')


@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html', recommendation_message="No prediction yet.", treatments=[])


if __name__ == '__main__':
    app.run(debug=True)