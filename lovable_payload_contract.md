# Sell Smart /predict contract

POST /predict
Content-Type: application/json

{
  "current_price": 30000,
  "quantity": 100,
  "region": 1,
  "price_features": {
    "...": "Use the exact Model 1 feature names from the artifact"
  },
  "transport_features": {
    "...": "Use the exact Model 3 feature names from the artifact"
  },
  "transport_cost_now": null,
  "transport_cost_later": null,
  "storage_cost_per_unit": 0
}

The authoritative decision-system bundle is:
artifacts/Sell_Smart_AI_Decision_System_FINAL.joblib

The deployed runtime requires:
scikit-learn==1.6.1

Do not send GPS coordinates into the model unless the application feature contract explicitly calls for them. Region/Zone/Woreda mapping must be handled by the application layer.
