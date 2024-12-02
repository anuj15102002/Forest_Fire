from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained machine learning model (pickled)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    """
    This function serves the homepage of the application.
    It simply renders the 'forest_fire.html' template.
    """
    return render_template("forest_fire.html")

@app.route('/predict', methods=['POST'])
def predict():
    """
    This function handles prediction requests.
    It receives input data from the user, makes a prediction using the model,
    and returns the result to the user.
    """
    try:
        # Extract input values from the form and convert them to floats
        input_features = [float(x) for x in request.form.values()]
        
        # Convert input features into a format suitable for prediction (numpy array)
        final_input = [np.array(input_features)]
        
        # Make a prediction using the trained model (predict probability of fire)
        prediction = model.predict_proba(final_input)
        
        # Format the predicted probability to 2 decimal places
        fire_probability = '{0:.{1}f}'.format(prediction[0][1], 2)

        # Check if the probability of fire is greater than 0.5 (threshold for danger)
        if float(fire_probability) > 0.5:
            return render_template('forest_fire.html', 
                                   pred=f'Your Forest is in Danger.\nProbability of fire occurring is {fire_probability}', 
                                   bhai="kuch karna hain iska ab?")
        else:
            return render_template('forest_fire.html', 
                                   pred=f'Your Forest is safe.\nProbability of fire occurring is {fire_probability}', 
                                   bhai="Your Forest is Safe for now")

    except Exception as e:
        # Handle errors gracefully by returning a message
        return f"An error occurred: {e}", 500

# Run the application in debug mode for development
if __name__ == '__main__':
    app.run(debug=True)
