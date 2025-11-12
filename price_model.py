import joblib
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, timedelta
import os
import sys

# Import RobustLabelEncoder from utils
try:
    from utils import RobustLabelEncoder
except ImportError:
    from sklearn.preprocessing import LabelEncoder
    class RobustLabelEncoder(LabelEncoder):
        pass

class PricePredictionModel:
    """Handles crop price predictions"""
    
    def __init__(self, model_path='models/crop_price_prediction_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
        # MSP fallback prices (2025 data)
        self.msp_prices = {
            'Rice': 2320, 'Wheat': 2425, 'Cotton': 6969, 
            'Tomato': 2200, 'Onion': 1800, 'Potato': 1400, 
            'Maize': 2090, 'Soybean': 4600, 'Sugarcane': 340,
            'Groundnut': 6377, 'Bajra': 2625, 'Jowar': 3570
        }
    
    def load_model(self):
        """Load the trained price prediction model"""
        try:
            if os.path.exists(self.model_path):
                import __main__
                __main__.RobustLabelEncoder = RobustLabelEncoder
                
                self.model = joblib.load(self.model_path)
                print("✅ Price prediction model loaded")
            else:
                print("⚠️ Price model not found, using MSP fallback")
        except Exception as e:
            print(f"⚠️ Error loading price model: {e}")
            print("   Using MSP-based predictions")
            self.model = None
    
    def predict(self, crop_name: str, location: Dict, soil_characteristics: Dict, 
                harvest_date: str = None) -> Dict:
        """Predict crop price"""
        try:
            if self.model:
                return self._predict_with_ml(crop_name, location, soil_characteristics, harvest_date)
            else:
                return self._predict_with_msp(crop_name, soil_characteristics, harvest_date)
        except Exception as e:
            print(f"⚠️ Price prediction error: {e}")
            return self._predict_with_msp(crop_name, soil_characteristics, harvest_date)
    
    def _predict_with_ml(self, crop_name: str, location: Dict, 
                        soil_characteristics: Dict, harvest_date: str) -> Dict:
        """ML-based price prediction"""
        try:
            models_dict = self.model['models']
            encoders_dict = self.model['encoders']
            scaler = self.model['scaler']
            feature_columns = self.model['feature_columns']
            reference_data = self.model['reference_data']
            
            state = location.get('state', 'Maharashtra')
            district = location.get('district', 'Unknown')
            
            if not harvest_date:
                growth_periods = {
                    'Rice': 120, 'Wheat': 120, 'Cotton': 180, 
                    'Onion': 120, 'Potato': 90, 'Tomato': 90, 'Maize': 120
                }
                days = growth_periods.get(crop_name, 120)
                harvest_dt = datetime.now() + timedelta(days=days)
            else:
                harvest_dt = pd.to_datetime(harvest_date)
            
            # Encode features safely
            def safe_encode(encoder, value, default=-1):
                try:
                    if hasattr(encoder, 'transform'):
                        return encoder.transform([str(value)])[0]
                    else:
                        return default
                except:
                    return default
            
            state_encoded = safe_encode(encoders_dict.get('STATE'), state)
            district_encoded = safe_encode(encoders_dict.get('District_Name'), district)
            commodity_encoded = safe_encode(encoders_dict.get('Commodity'), crop_name)
            
            # Get historical context
            similar_records = reference_data[reference_data['Commodity'] == crop_name]
            if len(similar_records) < 5:
                similar_records = reference_data
            
            avg_min = similar_records['Min_Price'].median()
            avg_max = similar_records['Max_Price'].median()
            
            # FIX: Use weekday() instead of dayofweek
            feature_dict = {
                'STATE_encoded': state_encoded,
                'District_Name_encoded': district_encoded,
                'Commodity_encoded': commodity_encoded,
                'Year': harvest_dt.year,
                'Month': harvest_dt.month,
                'Day': harvest_dt.day,
                'DayOfWeek': harvest_dt.weekday(),  # FIX: changed from dayofweek
                'Quarter': (harvest_dt.month - 1) // 3 + 1,  # Calculate quarter
                'Min_Price': avg_min,
                'Max_Price': avg_max,
                'Price_Range': avg_max - avg_min,
                'Price_Volatility': (avg_max - avg_min) / (avg_min + 0.001)
            }
            
            prediction_input = pd.DataFrame([
                {col: feature_dict.get(col, 0) for col in feature_columns}
            ])
            
            # Ensemble predictions
            predictions = {}
            for model_name, model in models_dict.items():
                try:
                    if model_name == 'Linear Regression':
                        pred_scaled = scaler.transform(prediction_input)
                        pred_price = model.predict(pred_scaled)[0]
                    else:
                        pred_price = model.predict(prediction_input)[0]
                    predictions[model_name] = max(pred_price, avg_min)
                except Exception as e:
                    predictions[model_name] = avg_min * 1.1
            
            # Weighted ensemble
            weights = {'Random Forest': 0.5, 'Gradient Boosting': 0.3, 'Linear Regression': 0.2}
            ensemble_price = sum(predictions[name] * weights.get(name, 0.1) 
                               for name in predictions) / sum(weights.get(name, 0.1) 
                               for name in predictions)
            
            # Soil quality adjustment
            soil_multiplier = self._calculate_soil_quality_multiplier(soil_characteristics)
            adjusted_price = int(ensemble_price * soil_multiplier)
            
            return {
                "predicted_price_per_quintal": adjusted_price,
                "price_range": {
                    "min": int(adjusted_price * 0.85),
                    "max": int(adjusted_price * 1.15)
                },
                "harvest_date": harvest_dt.strftime('%Y-%m-%d'),
                "confidence_level": "High - ML Model",
                "model_used": "Trained ML Model",
                "individual_predictions": {k: round(v, 2) for k, v in predictions.items()},
                "soil_quality_factor": f"{soil_multiplier:.3f}x"
            }
        except Exception as e:
            print(f"⚠️ ML prediction failed: {e}, falling back to MSP")
            return self._predict_with_msp(crop_name, soil_characteristics, harvest_date)
    
    def _predict_with_msp(self, crop_name: str, soil_characteristics: Dict, 
                         harvest_date: str) -> Dict:
        """MSP-based fallback prediction"""
        base_price = self.msp_prices.get(crop_name, 2000)
        
        if not harvest_date:
            harvest_date = (datetime.now() + timedelta(days=120)).strftime('%Y-%m-%d')
        
        soil_multiplier = self._calculate_soil_quality_multiplier(soil_characteristics)
        predicted_price = int(base_price * soil_multiplier)
        
        return {
            "predicted_price_per_quintal": predicted_price,
            "price_range": {
                "min": int(predicted_price * 0.85),
                "max": int(predicted_price * 1.15)
            },
            "harvest_date": harvest_date,
            "confidence_level": "Medium - MSP Based",
            "model_used": "MSP Fallback",
            "base_msp": base_price,
            "soil_quality_factor": f"{soil_multiplier:.3f}x"
        }
    
    def _calculate_soil_quality_multiplier(self, soil_characteristics: Dict) -> float:
        """Calculate soil quality adjustment factor"""
        multiplier = 1.0
        
        try:
            soil_data = soil_characteristics.get("validated", 
                                                soil_characteristics.get("predicted", {}))
            
            n_level = soil_data.get("N_level", soil_data.get("Nlevel"))
            if n_level == "High": multiplier += 0.05
            elif n_level == "Low": multiplier -= 0.03
            
            p_level = soil_data.get("P_level", soil_data.get("Plevel"))
            if p_level == "High": multiplier += 0.03
            elif p_level == "Low": multiplier -= 0.02
            
            k_level = soil_data.get("K_level", soil_data.get("Klevel"))
            if k_level == "High": multiplier += 0.03
            elif k_level == "Low": multiplier -= 0.02
            
            ph_level = soil_data.get("pH_level", soil_data.get("pHlevel"))
            if ph_level == "Neutral": multiplier += 0.02
        except:
            pass
        
        return max(0.85, min(1.25, multiplier))
