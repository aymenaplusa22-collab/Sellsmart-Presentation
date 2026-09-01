# Fast deployment

## GitHub
Create a private repository and upload this directory exactly as provided.

## Render
Create a new Web Service from the GitHub repository and choose **Docker**. The included `Dockerfile` and `render.yaml` configure the container. Render will generate `SESSION_SECRET` automatically when the blueprint is used.

After deployment, copy the public HTTPS URL and test:
- `/api/health`
- `/docs`

The website is served from `/` by the same FastAPI process, so no separate frontend host is required.

## Important
- Keep the repository private because it contains the model artifact.
- Keep `scikit-learn==1.6.1` exactly.
- Do not replace or retrain the `.joblib` artifact.
