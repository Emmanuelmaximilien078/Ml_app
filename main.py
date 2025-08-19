from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import joblib
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Autoriser le frontend Streamlit (ou Next.js local)
origins = [
    "http://localhost:3000",  # Next.js
    "http://localhost:8501",  # Streamlit par défaut
    # "https://ton-domaine.com" # prod
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle et le scaler
model_path = "random_Forest_model.sav"
scaler_path = "Robust_scaler.sav"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    raise RuntimeError(f"Erreur de chargement du modèle ou scaler : {e}")


@app.post("/predict")
async def predict_from_file(file: UploadFile = File(...)):
    try:
        if file.content_type != "text/csv":
            raise HTTPException(status_code=400, detail="Le fichier doit être au format CSV.")

        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))

        required_columns = ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"Colonnes manquantes. Requises : {required_columns}")

        features = df[required_columns]
        scaled_data = scaler.transform(features)

        predictions = model.predict(scaled_data)               # 0 ou 1
        probabilities = model.predict_proba(scaled_data)[:, 1] # Proba "vrai"

        df["prediction"] = predictions
        df["probability"] = (probabilities * 100).round(2)
        df["result_text"] = df.apply(
            lambda row: f"Vrai à {row['probability']}%" if row["prediction"] == 1
                        else f"Faux à {100 - row['probability']}%",
            axis=1
        )

        stats = {
            "total": len(df),
            "vrais": int((df["prediction"] == 1).sum()),
            "faux": int((df["prediction"] == 0).sum()),
            "pourcentage_vrais": round((df["prediction"] == 1).mean() * 100, 2),
            "pourcentage_faux": round((df["prediction"] == 0).mean() * 100, 2)
        }

        return {
            "table_predictions": df.to_dict(orient="records"),
            "statistiques": stats
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement : {e}")
