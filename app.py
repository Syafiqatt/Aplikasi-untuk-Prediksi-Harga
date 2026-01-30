from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model sekali saja saat aplikasi start
with open("pricing_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    price = None

    if request.method == "POST":
        try:
            # Pastikan konversi float berjalan lancar
            demand = float(request.form["demand"])
            stock = float(request.form["stock"])
            competition = float(request.form["competition"])

            # Prediksi
            prediction = model.predict([[demand, stock, competition]])[0]
            
            # Formatting harga (2 desimal)
            price = f"{prediction:.2f}"
            
        except ValueError:
            price = "Error: Input harus berupa angka valid"
        except Exception as e:
            price = f"Error: {str(e)}"

    return render_template("index.html", price=price)

if __name__ == "__main__":
    app.run(debug=True)