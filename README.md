# Sell Smart — Presentation App

A self-contained FastAPI web app for the Thursday presentation. It keeps the authoritative Sell Smart decision artifact on the server and provides a mobile-first UI for farmer accounts, saved location, prediction input, and prediction results.

## AI source of truth
- `data/Sell_Smart_AI_Decision_System_FINAL.joblib`
- Requires `scikit-learn==1.6.1`
- Do not retrain, replace, simplify, or modify the artifact.
- The backend uses the same decision-layer calculations supplied with the deployment bundle.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://127.0.0.1:8000

## Deploy
The included Dockerfile is ready for a Python-capable host such as Render or Railway. The site and API are served by the same FastAPI service, so there is no separate frontend host or Lovable dependency.

## Presentation scope
- Sign up / login / logout
- Saved Region / Zone / Woreda
- Dashboard greeting
- Real prediction flow using the authoritative artifact
- Prediction result card
- Profile editing
- Responsive mobile/desktop UI

Password reset/email delivery, production-grade email verification, and GPS are intentionally not on the critical presentation path.
