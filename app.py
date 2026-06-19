from flask import Flask, jsonify, render_template, request
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Cargar el modelo exportado desde la libreta
ruta_modelo = os.path.join(
    os.path.dirname(__file__),
    "modelo_peso_peces.pkl"
)

paquete_modelo = joblib.load(ruta_modelo)

modelo = paquete_modelo["modelo"]
escalador = paquete_modelo["escalador"]
columnas_numericas = paquete_modelo["columnas_numericas"]
columnas_dummy = paquete_modelo["columnas_dummy"]
caracteristicas = paquete_modelo["caracteristicas"]


@app.route("/")
def inicio():
    return render_template("formulario.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        length3 = float(request.form["length3"])
        width = float(request.form["width"])
        species = request.form["species"]

        if length3 <= 0 or width <= 0:
            return jsonify({
                "error": "Las medidas deben ser mayores que cero."
            }), 400

        # Crear las columnas numéricas en el mismo orden del entrenamiento
        datos_numericos = pd.DataFrame(
            [[0.0] * len(columnas_numericas)],
            columns=columnas_numericas
        )

        datos_numericos.loc[0, "Length3"] = length3
        datos_numericos.loc[0, "Width"] = width

        # Aplicar el mismo escalado utilizado en la libreta
        datos_escalados = pd.DataFrame(
            escalador.transform(datos_numericos),
            columns=columnas_numericas
        )

        # Crear las columnas de especie
        datos_especies = pd.DataFrame(
            [[0] * len(columnas_dummy)],
            columns=columnas_dummy
        )

        columna_especie = f"Species_{species}"

        if columna_especie in datos_especies.columns:
            datos_especies.loc[0, columna_especie] = 1

        # Unir datos numéricos y especie
        datos_completos = pd.concat(
            [datos_escalados, datos_especies],
            axis=1
        )

        # Usar únicamente las características del modelo final
        datos_finales = datos_completos[caracteristicas]

        prediccion = modelo.predict(datos_finales)

        peso_estimado = round(
            max(0.0, float(prediccion[0])),
            2
        )

        return jsonify({"peso": peso_estimado})

    except ValueError:
        return jsonify({
            "error": "Introduce valores numéricos válidos."
        }), 400

    except Exception as error:
        print("Error:", error)
        return jsonify({
            "error": "No fue posible realizar la predicción."
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
