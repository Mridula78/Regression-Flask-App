from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

# Data for the first model (Polynomial Regression)
x1 = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
y1 = [16, 25, 36, 49, 64, 81, 100]

# Data for the second model (Linear Regression)
x2 = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0]]
y2 = [8, 10, 12, 14, 16, 18, 20, 22]

# First model: Polynomial Regression
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression(),
)
polynomial_regression.fit(x1, y1)

# Second model: Linear Regression
lin_reg_split = LinearRegression()
lin_reg_split.fit(x2, y2)

@app.route('/', methods=['GET', 'POST'])
def home():
    poly_prediction = None
    lin_reg_prediction = None
    input_value = None
    input_value_2 = None

    if request.method == 'POST':
        # Get the user input
        input_value = request.form.get('input_value')
        input_value_2 = request.form.get('input_value_2')

        try:
            # Convert input to float
            input_value_float = float(input_value)
            input_value_2_float = float(input_value_2)

            # Polynomial Regression Prediction
            poly_prediction = polynomial_regression.predict([[input_value_float]])[0]

            # Linear Regression Prediction
            lin_reg_prediction = lin_reg_split.predict([[input_value_2_float]])[0]

            # Format predictions to two decimal places
            poly_prediction = f"{poly_prediction:.2f}"
            lin_reg_prediction = f"{lin_reg_prediction:.2f}"

        except ValueError:
            input_value = "Invalid input! Please enter numeric values."

    return render_template('index.html', 
                           poly_prediction=poly_prediction,
                           lin_reg_prediction=lin_reg_prediction,
                           input_value=input_value,
                           input_value_2=input_value_2)

if __name__ == '__main__':
    app.run(debug=True)
