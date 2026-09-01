# Sell Smart presentation deployment checklist

- Python runtime: 3.11 (Dockerfile)
- scikit-learn: 1.6.1 (pinned)
- Authoritative artifact: data/Sell_Smart_AI_Decision_System_FINAL.joblib
- Model artifact is server-side only.
- `/api/health` should report artifact version FINAL_V1, threshold 2.0, sklearn 1.6.1.
- `/api/me` returns only safe profile fields; password hash/salt are never returned.
- `COOKIE_SECURE=1` for HTTPS production.
- For demo, SQLite is used. Ephemeral hosts may reset SQLite data after redeploy/restart; use a durable database for production.
- Test `/api/health`, sign-up, login, prediction, profile save, and logout before presenting.
