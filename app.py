#This file uses the pickled model to make predictions using user inputted date on the web application
#This file in combination with the index.html file serves a web application that can be interacted with in your preferred browser

# Necessary imports to run web app
from flask import Flask, render_template, request
import pickle
import pandas as pd

#Load the pickled model
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

#Numerical features used for predictions
important_features = [
    "wheel-base", "length", "width", "curb-weight", "engine-size",
    "horsepower", "city-mpg", "highway-mpg"
]

# Styling info for the form
NUMERIC_INFO = {
    "wheel-base":   {"label": "Wheel base",      "unit": "inches"},
    "length":       {"label": "Length",          "unit": "inches"},
    "width":        {"label": "Width",           "unit": "inches"},
    "curb-weight":  {"label": "Curb weight",     "unit": "lbs"},
    "engine-size":  {"label": "Engine size",     "unit": "cc"},
    "horsepower":   {"label": "Horsepower",      "unit": "hp"},
    "city-mpg":     {"label": "City Miles Per Gallon",    "unit": "mpg"},
    "highway-mpg":  {"label": "Highway Miles Per Gallon", "unit": "mpg"},
}

# Categorical features used to make prediction
categorical_features = [
    "make", "fuel-type", "aspiration", "num-of-doors",
    "body-style", "drive-wheels", "engine-location", "engine-type",
    "num-of-cylinders", "fuel-system"
]

# Styling info for the form
CATEGORICAL_INFO = {
    "make":            "Make",
    "fuel-type":       "Fuel type",
    "aspiration":      "Aspiration",
    "num-of-doors":    "Number of doors",
    "body-style":      "Body style",
    "drive-wheels":    "Drive wheels",
    "engine-location": "Engine location",
    "engine-type":     "Engine type",
    "num-of-cylinders":"Number of cylinders",
    "fuel-system":     "Fuel system",
}

# All features used to make predictions numerical and categorical
selected_features = important_features + categorical_features

# Default values if nothing is entered in the form by the user
FEATURE_DEFAULTS = {
    "wheel-base": 95.0,
    "length": 175.0,
    "width": 66.0,
    "curb-weight": 2500.0,
    "engine-size": 130.0,
    "horsepower": 100.0,
    "city-mpg": 25.0,
    "highway-mpg": 30.0,

    "make": "toyota",
    "fuel-type": "gas",
    "aspiration": "std",
    "num-of-doors": "four",
    "body-style": "sedan",
    "drive-wheels": "fwd",
    "engine-location": "front",
    "engine-type": "ohc",
    "num-of-cylinders": "four",
    "fuel-system": "mpfi",
}

# Choices provided to the user so the categorical features can be a dropdown
CATEGORICAL_CHOICES = {
    "make": [
        "Alfa-Romero", "Audi", "BMW", "Chevrolet", "Dodge", "Honda",
        "Isuzu", "Jaguar", "Mazda", "Mercedes-Benz", "Mitsubishi",
        "Nissan", "Peugot", "Plymouth", "Porsche", "Renault", "Saab",
        "Subaru", "Toyota", "Volkswagen", "Volvo"
    ],
    "fuel-type": ["Gas", "Diesel"],
    "aspiration": ["STD", "Turbo"],
    "num-of-doors": ["Two", "Four"],
    "body-style": ["Sedan", "Hatchback", "Wagon", "Hardtop", "Convertible"],
    "drive-wheels": ["RWD", "RWD", "4WD"],
    "engine-location": ["Front", "Rear"],
    "engine-type": ["OHC", "OHCF", "OHCV", "L", "Rotor", "DOHC"],
    "num-of-cylinders": ["Two", "Three", "Four", "Five", "Six", "Eight", "Twelve"],
    "fuel-system": ["MPFI", "2BBL", "1BBL", "4BBL", "IDI", "SPDI", "SPFI", "MFI"],
}

# Create application object
app = Flask(__name__)


# Web application routing and prediction
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    errors = []
    form_values = {f: "" for f in selected_features}

    # Executed when a form is submitted
    if request.method == "POST":
        input_data = {}

        # Get user inputted data from form allow blank inputs
        for feat in selected_features:
            raw_value = request.form.get(feat, "")
            form_values[feat] = raw_value

            # If the user left it blank, use a default
            if raw_value is None or raw_value.strip() == "":
                default_value = FEATURE_DEFAULTS.get(feat)

                # numeric vs categorical
                if feat in important_features:
                    input_data[feat] = float(default_value)
                else:
                    input_data[feat] = str(default_value)
                continue

            # If they did type something
            if feat in important_features:
                # numeric feature: convert to float on error use default value
                try:
                    input_data[feat] = float(raw_value)
                except (TypeError, ValueError):
                    errors.append(f"Invalid numeric value for '{feat}'. "
                                  f"Using default instead.")
                    default_value = FEATURE_DEFAULTS.get(feat)
                    input_data[feat] = float(default_value)
            else:
                # categorical: use downdown selection
                input_data[feat] = raw_value

        # Get prediction using user inputs on the pickled model
        if not errors:
            X_new = pd.DataFrame([input_data])
            y_pred = model.predict(X_new)[0]
            prediction = round(float(y_pred), 2)

    return render_template(
        "index.html",
        prediction=prediction,
        errors=errors,
        form_values=form_values,
        numeric_features=important_features,
        categorical_features=categorical_features,
        categorical_choices=CATEGORICAL_CHOICES,
        numeric_info=NUMERIC_INFO,
        categorical_info=CATEGORICAL_INFO,
        feature_defaults=FEATURE_DEFAULTS
    )



if __name__ == "__main__":
    app.run(debug=True)
