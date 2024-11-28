from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def calculator():
    return render_template("calculator.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        # Obtener los números del formulario
        num1 = float(request.form.get("num1"))
        num2 = float(request.form.get("num2"))
    except (TypeError, ValueError):
        return render_template("calculator.html", error="Por favor ingresa dos números válidos.")
    
    # Calcular la suma
    result = num1 + num2
    return render_template("calculator.html", result=result, num1=num1, num2=num2)

