# SME Success Predictor API

A Flask REST API with Swagger UI for predicting Small and Medium Enterprise (SME) business success in Rwanda using machine learning.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Navigate to the API directory:**
   ```bash
   cd api
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API:**
   ```bash
   python app.py
   ```

4. **Access the API:**
   - **Swagger UI (Interactive API Documentation):** http://localhost:5000/swagger/
   - **Home Page:** http://localhost:5000/
   - **Health Check:** http://localhost:5000/health

## 📊 API Endpoints

### 🔮 Single Prediction
**POST** `/api/v1/predictions/single`

Predict success for a single SME business.

**Example Request:**
```json
{
  "industry_technology": 1,
  "industry_manufacturing": 0,
  "industry_services": 0,
  "industry_agriculture": 0,
  "industry_trade": 0,
  "capital_amount": 2500000.0,
  "business_age_years": 3,
  "owner_age": 35,
  "owner_education_secondary": 0,
  "owner_education_university": 1,
  "owner_experience_years": 8,
  "location_urban": 1,
  "gender_male": 1,
  "num_employees": 15,
  "growth_indicator": 0.25
}
```

**Example Response:**
```json
{
  "prediction": "Operating",
  "success_probability": 0.872,
  "risk_level": "Low Risk",
  "confidence": 0.872,
  "recommendations": [
    "Business shows strong potential - maintain current strategy",
    "Consider expansion opportunities and market diversification"
  ],
  "timestamp": "2025-10-07 15:30:00",
  "model_info": {
    "model_type": "LogisticRegression",
    "features_count": 19,
    "version": "1.0"
  }
}
```

### 📊 Batch Prediction
**POST** `/api/v1/predictions/batch`

Predict success for multiple SMEs in a single request.

**Example Request:**
```json
[
  {
    "industry_technology": 1,
    "capital_amount": 2500000.0,
    "business_age_years": 3,
    // ... other fields
  },
  {
    "industry_manufacturing": 1,
    "capital_amount": 1500000.0,
    "business_age_years": 5,
    // ... other fields
  }
]
```

### ℹ️ Model Information
**GET** `/api/v1/predictions/model-info`

Get detailed information about the loaded model.

### ❤️ Health Check
**GET** `/health`

Check API health status.

## 🏗️ Input Fields Reference

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `industry_technology` | Integer (0/1) | Technology industry flag | 1 |
| `industry_manufacturing` | Integer (0/1) | Manufacturing industry flag | 0 |
| `industry_services` | Integer (0/1) | Services industry flag | 0 |
| `industry_agriculture` | Integer (0/1) | Agriculture industry flag | 0 |
| `industry_trade` | Integer (0/1) | Trade industry flag | 0 |
| `capital_amount` | Float | Initial capital in RWF | 2500000.0 |
| `business_age_years` | Integer | Business age in years | 3 |
| `owner_age` | Integer | Owner age | 35 |
| `owner_education_secondary` | Integer (0/1) | Secondary education flag | 0 |
| `owner_education_university` | Integer (0/1) | University education flag | 1 |
| `owner_experience_years` | Integer | Owner experience in years | 8 |
| `location_urban` | Integer (0/1) | Urban location flag | 1 |
| `gender_male` | Integer (0/1) | Male gender flag | 1 |
| `num_employees` | Integer | Number of employees | 15 |
| `growth_indicator` | Float | Growth indicator (-0.2 to 0.5) | 0.25 |

## 🎯 Model Performance

- **Algorithm:** Logistic Regression
- **Accuracy:** 84.5%
- **Features:** 19 engineered features
- **Dataset:** Rwanda SME data (1,000 records)
- **Target Classes:** Operating, Closed

## 🧪 Testing Examples

### Using curl:

```bash
# Single prediction
curl -X POST "http://localhost:5000/api/v1/predictions/single" \
     -H "Content-Type: application/json" \
     -d '{
       "industry_technology": 1,
       "industry_manufacturing": 0,
       "industry_services": 0,
       "industry_agriculture": 0,
       "industry_trade": 0,
       "capital_amount": 2500000.0,
       "business_age_years": 3,
       "owner_age": 35,
       "owner_education_secondary": 0,
       "owner_education_university": 1,
       "owner_experience_years": 8,
       "location_urban": 1,
       "gender_male": 1,
       "num_employees": 15,
       "growth_indicator": 0.25
     }'

# Health check
curl http://localhost:5000/health
```

### Using Python requests:

```python
import requests

# Single prediction
url = "http://localhost:5000/api/v1/predictions/single"
data = {
    "industry_technology": 1,
    "capital_amount": 2500000.0,
    "business_age_years": 3,
    "owner_age": 35,
    "owner_education_university": 1,
    "location_urban": 1,
    "gender_male": 1,
    "num_employees": 15,
    "growth_indicator": 0.25,
    # ... include all required fields
}

response = requests.post(url, json=data)
print(response.json())
```

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Set to `development` for debug mode
- `FLASK_DEBUG`: Set to `1` for debug mode
- `PORT`: API port (default: 5000)

### Model Files
- The API expects the trained model file at: `../models/best_model_logistic_regression.pkl`
- Ensure the model was trained using the ML pipeline notebook

## 🏭 Production Deployment

For production deployment, consider:

1. **Using gunicorn:**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Environment variables:**
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=0
   ```

3. **Reverse proxy with nginx**
4. **SSL/TLS certificates**
5. **Monitoring and logging**

## 🐛 Troubleshooting

### Common Issues:

1. **Model not found error:**
   - Ensure the model file exists at `../models/best_model_logistic_regression.pkl`
   - Run the ML pipeline notebook to generate the model

2. **Import errors:**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Port already in use:**
   - Change the port in `app.py` or kill the existing process

### Debug Mode:
Set `debug=True` in `app.run()` for detailed error messages.

## 📈 Monitoring

The API includes:
- Health check endpoint (`/health`)
- Structured logging
- Error handling with proper HTTP status codes
- Request/response validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of an academic capstone project for SME success prediction in Rwanda.