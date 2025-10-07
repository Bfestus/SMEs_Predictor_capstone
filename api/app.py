"""
SME Success Predictor API
Flask application with Swagger UI for testing the best trained model

This API provides endpoints to predict SME business success using the trained Logistic Regression model.
"""

from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields, Namespace
import joblib
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
import traceback

# Add the parent directory to Python path to import from notebooks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['RESTX_VALIDATE'] = True

# Initialize Flask-RESTX for Swagger documentation
api = Api(
    app,
    version='1.0',
    title='SME Success Predictor API',
    description='API for predicting Small and Medium Enterprise (SME) business success in Rwanda',
    doc='/swagger/',
    prefix='/api/v1'
)

# Create namespace for predictions
ns_predict = Namespace('predictions', description='SME Success Prediction Operations')
api.add_namespace(ns_predict, path='/api/v1/predictions')

# Global variables for model and feature names
model = None
feature_names = None
label_encoder = None

def load_model():
    """Load the best trained model and feature information"""
    global model, feature_names, label_encoder
    
    try:
        # Define paths
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        model_path = os.path.join(models_dir, 'best_model_logistic_regression.pkl')
        
        # Load the model
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from: {model_path}")
        else:
            logger.error(f"Model file not found at: {model_path}")
            return False
        
        # Define feature names based on the training pipeline
        feature_names = [
            'Business_Sector_Encoded', 'Business_Subsector_Encoded', 'Business_Model_Encoded',
            'Ownership_Type_Encoded', 'Location_Encoded', 'Owner_Gender_Encoded',
            'Business_Type_Encoded', 'Age_Category_Encoded', 'Capital_Size_Encoded',
            'Employee_Size_Encoded', 'Owner_Age_Category_Encoded', 'Registration_Year_Scaled',
            'Duration_Operation_Scaled', 'Business_Age_Scaled', 'Initial_Capital_Scaled',
            'Num_Employees_Scaled', 'Owner_Age_Scaled', 'Growth_Indicator_Scaled',
            'Capital_per_Employee_Scaled'
        ]
        
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Features expected: {len(feature_names)}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        traceback.print_exc()
        return False

# Load model on startup
if not load_model():
    logger.error("Failed to load model on startup")

# Define input model for Swagger documentation
sme_input_model = api.model('SMEInput', {
    'industry_technology': fields.Integer(required=True, description='Technology industry (0 or 1)', example=1),
    'industry_manufacturing': fields.Integer(required=True, description='Manufacturing industry (0 or 1)', example=0),
    'industry_services': fields.Integer(required=True, description='Services industry (0 or 1)', example=0),
    'industry_agriculture': fields.Integer(required=True, description='Agriculture industry (0 or 1)', example=0),
    'industry_trade': fields.Integer(required=True, description='Trade industry (0 or 1)', example=0),
    'capital_amount': fields.Float(required=True, description='Initial capital in RWF', example=2500000.0),
    'business_age_years': fields.Integer(required=True, description='Business age in years', example=3),
    'owner_age': fields.Integer(required=True, description='Owner age', example=35),
    'owner_education_secondary': fields.Integer(required=True, description='Secondary education (0 or 1)', example=0),
    'owner_education_university': fields.Integer(required=True, description='University education (0 or 1)', example=1),
    'owner_experience_years': fields.Integer(required=True, description='Owner experience in years', example=8),
    'location_urban': fields.Integer(required=True, description='Urban location (0 or 1)', example=1),
    'gender_male': fields.Integer(required=True, description='Male gender (0 or 1)', example=1),
    'num_employees': fields.Integer(required=True, description='Number of employees', example=15),
    'growth_indicator': fields.Float(required=True, description='Growth indicator (-0.2 to 0.5)', example=0.25)
})

# Define output model for Swagger documentation
prediction_output_model = api.model('PredictionOutput', {
    'prediction': fields.String(description='Predicted business status', example='Operating'),
    'success_probability': fields.Float(description='Probability of success (0-1)', example=0.872),
    'risk_level': fields.String(description='Risk assessment', example='Low Risk'),
    'confidence': fields.Float(description='Model confidence (0-1)', example=0.872),
    'recommendations': fields.List(fields.String, description='Business recommendations'),
    'timestamp': fields.String(description='Prediction timestamp', example='2025-10-07 15:30:00'),
    'model_info': fields.Raw(description='Model information')
})

def encode_sme_features(sme_data):
    """
    Convert user-friendly input to model features
    
    This function maps the simplified API input to the complex feature space
    that the trained model expects.
    """
    try:
        # Initialize feature vector with zeros
        features = np.zeros(len(feature_names))
        feature_dict = {}
        
        # Simple encoding mappings (these would ideally be loaded from training artifacts)
        
        # Industry encoding (one-hot)
        if sme_data.get('industry_technology', 0):
            feature_dict['Business_Sector_Encoded'] = 12  # Technology sector code
        elif sme_data.get('industry_manufacturing', 0):
            feature_dict['Business_Sector_Encoded'] = 6   # Manufacturing sector code
        elif sme_data.get('industry_services', 0):
            feature_dict['Business_Sector_Encoded'] = 9   # Services sector code
        elif sme_data.get('industry_agriculture', 0):
            feature_dict['Business_Sector_Encoded'] = 1   # Agriculture sector code
        elif sme_data.get('industry_trade', 0):
            feature_dict['Business_Sector_Encoded'] = 11  # Trade sector code
        else:
            feature_dict['Business_Sector_Encoded'] = 0   # Default/Other
        
        # Business model (simplified)
        feature_dict['Business_Model_Encoded'] = 1  # Default to hybrid model
        
        # Ownership type (simplified)
        feature_dict['Ownership_Type_Encoded'] = 1  # Default to sole proprietorship
        
        # Location
        feature_dict['Location_Encoded'] = sme_data.get('location_urban', 1)
        
        # Owner gender
        feature_dict['Owner_Gender_Encoded'] = sme_data.get('gender_male', 1)
        
        # Business type (based on capital and employees)
        capital = sme_data.get('capital_amount', 1000000)
        employees = sme_data.get('num_employees', 1)
        if capital > 5000000 or employees > 20:
            feature_dict['Business_Type_Encoded'] = 1  # SME
        elif capital > 1000000 or employees > 5:
            feature_dict['Business_Type_Encoded'] = 0  # Micro-enterprise
        else:
            feature_dict['Business_Type_Encoded'] = 2  # Startup
        
        # Age category
        business_age = sme_data.get('business_age_years', 1)
        if business_age <= 1:
            feature_dict['Age_Category_Encoded'] = 0  # New
        elif business_age <= 5:
            feature_dict['Age_Category_Encoded'] = 1  # Young
        elif business_age <= 10:
            feature_dict['Age_Category_Encoded'] = 2  # Mature
        else:
            feature_dict['Age_Category_Encoded'] = 3  # Established
        
        # Capital size
        if capital < 1000000:
            feature_dict['Capital_Size_Encoded'] = 0  # Small
        elif capital < 5000000:
            feature_dict['Capital_Size_Encoded'] = 1  # Medium
        elif capital < 10000000:
            feature_dict['Capital_Size_Encoded'] = 2  # Large
        else:
            feature_dict['Capital_Size_Encoded'] = 3  # Very Large
        
        # Employee size
        if employees <= 5:
            feature_dict['Employee_Size_Encoded'] = 0  # Small
        elif employees <= 20:
            feature_dict['Employee_Size_Encoded'] = 1  # Medium
        else:
            feature_dict['Employee_Size_Encoded'] = 2  # Large
        
        # Owner age category
        owner_age = sme_data.get('owner_age', 35)
        if owner_age < 30:
            feature_dict['Owner_Age_Category_Encoded'] = 0  # Young
        elif owner_age < 50:
            feature_dict['Owner_Age_Category_Encoded'] = 1  # Middle-aged
        else:
            feature_dict['Owner_Age_Category_Encoded'] = 2  # Senior
        
        # Scaled numerical features (simplified scaling)
        # These would ideally use the same scaler from training
        
        # Registration year (assuming recent years)
        current_year = 2025
        registration_year = current_year - business_age
        feature_dict['Registration_Year_Scaled'] = (registration_year - 2012.935) / 7.351518
        
        # Duration and age
        feature_dict['Duration_Operation_Scaled'] = (business_age - 12.065) / 7.351518
        feature_dict['Business_Age_Scaled'] = (business_age - 10.813) / 7.902815
        
        # Capital
        feature_dict['Initial_Capital_Scaled'] = (capital - 6045034) / 4627611
        
        # Employees
        feature_dict['Num_Employees_Scaled'] = (employees - 23.896) / 14.056058
        
        # Owner age
        feature_dict['Owner_Age_Scaled'] = (owner_age - 40.635) / 13.525632
        
        # Growth indicator
        growth = sme_data.get('growth_indicator', 0.1)
        feature_dict['Growth_Indicator_Scaled'] = (growth - 0.144120) / 0.201448
        
        # Capital per employee
        capital_per_employee = capital / max(employees, 1)
        # Estimate scaling based on typical values
        feature_dict['Capital_per_Employee_Scaled'] = (capital_per_employee - 300000) / 200000
        
        # Subsector (simplified)
        feature_dict['Business_Subsector_Encoded'] = feature_dict['Business_Sector_Encoded'] * 3  # Simple mapping
        
        # Map to feature vector
        for i, feature_name in enumerate(feature_names):
            features[i] = feature_dict.get(feature_name, 0)
        
        return features
        
    except Exception as e:
        logger.error(f"Error encoding features: {str(e)}")
        raise

def get_risk_level(success_prob):
    """Determine risk level based on success probability"""
    if success_prob >= 0.8:
        return "Low Risk"
    elif success_prob >= 0.6:
        return "Medium Risk"
    elif success_prob >= 0.4:
        return "High Risk"
    else:
        return "Very High Risk"

def get_recommendations(success_prob, sme_data):
    """Generate actionable recommendations"""
    recommendations = []
    
    if success_prob < 0.6:
        recommendations.append("Consider business strategy review and market analysis")
        recommendations.append("Explore additional funding sources or cost optimization")
        
    if success_prob < 0.4:
        recommendations.append("Urgent: Seek business mentorship and advisory support")
        recommendations.append("Focus on core competencies and operational efficiency")
        
    if sme_data.get('capital_amount', 0) < 1000000:
        recommendations.append("Consider increasing initial capital investment")
        
    if sme_data.get('owner_education_university', 0) == 0:
        recommendations.append("Consider upgrading skills through training programs")
        
    if len(recommendations) == 0:
        recommendations.append("Business shows strong potential - maintain current strategy")
        recommendations.append("Consider expansion opportunities and market diversification")
        
    return recommendations

@ns_predict.route('/single')
class SinglePrediction(Resource):
    @ns_predict.expect(sme_input_model)
    @ns_predict.marshal_with(prediction_output_model)
    @ns_predict.doc('predict_sme_success')
    def post(self):
        """
        Predict success for a single SME
        
        Provides prediction, probability, risk assessment, and recommendations
        for a single Small and Medium Enterprise based on business characteristics.
        """
        try:
            if model is None:
                return {'error': 'Model not loaded'}, 500
            
            # Get input data
            sme_data = request.json
            
            # Validate required fields
            required_fields = [
                'industry_technology', 'industry_manufacturing', 'industry_services',
                'industry_agriculture', 'industry_trade', 'capital_amount',
                'business_age_years', 'owner_age', 'owner_education_secondary',
                'owner_education_university', 'owner_experience_years',
                'location_urban', 'gender_male', 'num_employees', 'growth_indicator'
            ]
            
            missing_fields = [field for field in required_fields if field not in sme_data]
            if missing_fields:
                return {'error': f'Missing required fields: {missing_fields}'}, 400
            
            # Encode features
            features = encode_sme_features(sme_data)
            features_df = pd.DataFrame([features], columns=feature_names)
            
            # Make prediction
            prediction = model.predict(features_df)[0]
            probability = model.predict_proba(features_df)[0]
            
            # Format results
            result = {
                'prediction': 'Operating' if prediction == 1 else 'Closed',
                'success_probability': float(probability[1]),
                'risk_level': get_risk_level(probability[1]),
                'confidence': float(max(probability)),
                'recommendations': get_recommendations(probability[1], sme_data),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_info': {
                    'model_type': type(model).__name__,
                    'features_count': len(feature_names),
                    'version': '1.0'
                }
            }
            
            logger.info(f"Prediction made: {result['prediction']} ({result['success_probability']:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            traceback.print_exc()
            return {'error': f'Prediction failed: {str(e)}'}, 500

@ns_predict.route('/batch')
class BatchPrediction(Resource):
    @ns_predict.expect([sme_input_model])
    @ns_predict.marshal_with([prediction_output_model])
    @ns_predict.doc('predict_multiple_smes')
    def post(self):
        """
        Predict success for multiple SMEs
        
        Batch prediction endpoint for processing multiple SME requests at once.
        Input should be an array of SME data objects.
        """
        try:
            if model is None:
                return {'error': 'Model not loaded'}, 500
            
            # Get input data (should be a list)
            sme_list = request.json
            if not isinstance(sme_list, list):
                return {'error': 'Input should be an array of SME data objects'}, 400
            
            results = []
            for i, sme_data in enumerate(sme_list):
                try:
                    # Encode features
                    features = encode_sme_features(sme_data)
                    features_df = pd.DataFrame([features], columns=feature_names)
                    
                    # Make prediction
                    prediction = model.predict(features_df)[0]
                    probability = model.predict_proba(features_df)[0]
                    
                    # Format results
                    result = {
                        'sme_id': i + 1,
                        'prediction': 'Operating' if prediction == 1 else 'Closed',
                        'success_probability': float(probability[1]),
                        'risk_level': get_risk_level(probability[1]),
                        'confidence': float(max(probability)),
                        'recommendations': get_recommendations(probability[1], sme_data),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'model_info': {
                            'model_type': type(model).__name__,
                            'features_count': len(feature_names),
                            'version': '1.0'
                        }
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing SME {i+1}: {str(e)}")
                    results.append({
                        'sme_id': i + 1,
                        'error': f'Processing failed: {str(e)}'
                    })
            
            logger.info(f"Batch prediction completed for {len(sme_list)} SMEs")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            traceback.print_exc()
            return {'error': f'Batch prediction failed: {str(e)}'}, 500

@ns_predict.route('/model-info')
class ModelInfo(Resource):
    @ns_predict.doc('get_model_info')
    def get(self):
        """
        Get information about the loaded model
        
        Returns details about the currently loaded model including
        features, performance metrics, and model configuration.
        """
        try:
            if model is None:
                return {'error': 'Model not loaded'}, 500
            
            info = {
                'model_type': type(model).__name__,
                'model_parameters': model.get_params() if hasattr(model, 'get_params') else {},
                'features_count': len(feature_names),
                'feature_names': feature_names,
                'model_loaded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'api_version': '1.0',
                'description': 'SME Success Predictor trained on Rwanda SME dataset',
                'target_classes': ['Closed', 'Operating'],
                'performance_metrics': {
                    'accuracy': 0.845,
                    'model_file': 'best_model_logistic_regression.pkl'
                }
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Model info error: {str(e)}")
            return {'error': f'Failed to get model info: {str(e)}'}, 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_loaded': model is not None
    })

@app.route('/')
def home():
    """Redirect to Swagger documentation"""
    return """
    <h1>SME Success Predictor API</h1>
    <p>Welcome to the SME Success Predictor API!</p>
    <p><a href="/swagger/">View API Documentation (Swagger UI)</a></p>
    <p><a href="/health">Health Check</a></p>
    
    <h2>Quick Start</h2>
    <p>This API provides machine learning predictions for Small and Medium Enterprise (SME) business success in Rwanda.</p>
    
    <h3>Available Endpoints:</h3>
    <ul>
        <li><strong>POST /api/v1/predictions/single</strong> - Predict success for a single SME</li>
        <li><strong>POST /api/v1/predictions/batch</strong> - Predict success for multiple SMEs</li>
        <li><strong>GET /api/v1/predictions/model-info</strong> - Get model information</li>
        <li><strong>GET /health</strong> - Health check</li>
    </ul>
    
    <h3>Model Performance:</h3>
    <ul>
        <li>Algorithm: Logistic Regression</li>
        <li>Accuracy: 84.5%</li>
        <li>Features: 19 engineered features</li>
        <li>Dataset: Rwanda SME data (1,000 records)</li>
    </ul>
    """

if __name__ == '__main__':
    print("🚀 Starting SME Success Predictor API...")
    print("📊 Model loaded:", model is not None)
    print("🌐 Swagger UI available at: http://localhost:5000/swagger/")
    print("🏠 Home page at: http://localhost:5000/")
    print("❤️ Health check at: http://localhost:5000/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)