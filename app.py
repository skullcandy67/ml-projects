import pickle
from flask import Flask,render_template,request,flash,jsonify
import numpy as np

model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
    
app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and validate input
        float_features = [float(x) for x in request.form.values()]
        if len(float_features) != 4:
            flash('Fill all the fields.', 'warning')
            return render_template('index.html')

        # Convert input into array
        final_features = [np.array(float_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0], 3)

        return render_template('index.html', prediction_text=output)
    
    except ValueError:
        flash('Invalid input! Please enter numerical values.', 'danger')
        return render_template('index.html')

    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        return render_template('index.html')

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    app.run(debug=False)