from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os


app = Flask(__name__)


# Ruta donde se encuentra el archivo del modelo
ruta_modelo = os.path.join(
    os.path.dirname(__file__),
    "modelo_peso_peces.pkl"
)


# Cargar el paquete exportado desde el notebook
paquete_modelo = joblib.load(ruta_modelo)

modelo = paquete_modelo["modelo"]
escalador = paquete_modelo["escalador"]
columnas = paquete_modelo["columnas_originales"]
caracteristicas = paquete_modelo["caracteristicas"]


@app.route("/")
def inicio():
    return render_template("formulario.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Recibir las tres longitudes del formulario
        length1 = float(request.form["length1"])
        length2 = float(request.form["length2"])
        length3 = float(request.form["length3"])

        # Validar que las medidas sean mayores que cero
        if length1 <= 0 or length2 <= 0 or length3 <= 0:
            return jsonify({
                "error": "Las longitudes deben ser mayores que cero."
            }), 400

        # Crear un registro con todas las columnas usadas por el escalador
        datos = pd.DataFrame(
            [[0.0] * len(columnas)],
            columns=columnas
        )

        # Colocar las medidas ingresadas por el usuario
        datos.loc[0, "Length1"] = length1
        datos.loc[0, "Length2"] = length2
        datos.loc[0, "Length3"] = length3

        # Aplicar el mismo escalado utilizado en el entrenamiento
        datos_escalados = escalador.transform(datos)

        datos_escalados = pd.DataFrame(
            datos_escalados,
            columns=columnas
        )

        # Utilizar únicamente las características finales del modelo
        datos_finales = datos_escalados[caracteristicas]

        # Realizar la predicción
        prediccion = modelo.predict(datos_finales)

        peso_estimado = round(float(prediccion[0]), 2)

        return jsonify({
            "peso": peso_estimado
        })

    except ValueError:
        return jsonify({
            "error": "Debes introducir valores numéricos válidos."
        }), 400

    except Exception as error:
        print("Error:", error)

        return jsonify({
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)