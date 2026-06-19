from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

ruta_modelo = BASE_DIR / "modelo_peso_peces.pkl"

paquete_modelo = joblib.load(
    ruta_modelo
)

modelo = paquete_modelo["modelo"]

caracteristicas = paquete_modelo[
    "caracteristicas"
]

@app.route("/")
def inicio():
    return render_template(
        "formulario.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():
    try:
        length3 = float(
            request.form.get("length3", "")
        )

        width = float(
            request.form.get("width", "")
        )

        if length3 <= 0 or width <= 0:
            return jsonify({
                "error":
                    "Las medidas deben ser "
                    "mayores que cero."
            }), 400

        datos = pd.DataFrame([
            {
                "Width": width,
                "Length3": length3
            }
        ])

        datos_finales = datos[
            caracteristicas
        ]

        prediccion = modelo.predict(
            datos_finales
        )

        peso_estimado = round(
            max(
                0.0,
                float(prediccion[0])
            ),
            2
        )

        return jsonify({
            "peso": peso_estimado
        })

    except ValueError:
        return jsonify({
            "error":
                "Introduce valores numéricos "
                "válidos."
        }), 400

    except KeyError as error:
        print(
            "Falta una característica:",
            error
        )

        return jsonify({
            "error":
                "El modelo no tiene la estructura "
                "esperada."
        }), 500

    except Exception as error:
        print(
            "Error al predecir:",
            error
        )

        return jsonify({
            "error":
                "No fue posible realizar "
                "la predicción."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)