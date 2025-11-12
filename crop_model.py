import joblib
import pandas as pd
import numpy as np
from typing import Dict, List
import os

class CropRecommendationModel:
    """Handles crop recommendations based on location and soil data"""
    
    def __init__(self, model_path='models/crop_recommender_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained crop recommendation model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("✅ Crop recommendation model loaded")
            else:
                print("⚠️ Model not found, using fallback")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def predict(self, latitude: float, longitude: float, soil_data: Dict) -> List[Dict]:
        """Predict recommended crops"""
        try:
            if self.model and isinstance(self.model, dict) and 'model' in self.model:
                input_features = [
                    latitude, longitude, 
                    soil_data.get('N', 120),
                    soil_data.get('P', 40), 
                    soil_data.get('K', 150),
                    soil_data.get('ph', 7.0)
                ]
                
                input_df = pd.DataFrame(
                    [input_features], 
                    columns=['Latitude', 'Longitude', 'N', 'P', 'K', 'ph']
                )
                
                predictions = self.model['model'].predict(input_df)[0]
                
                if 'label_binarizer' in self.model:
                    recommended_crops = self.model['label_binarizer'].inverse_transform([predictions])[0]
                else:
                    recommended_crops = self._get_fallback_crops()
            else:
                recommended_crops = self._get_fallback_crops()
            
            # Format recommendations
            crop_list = []
            for i, crop in enumerate(recommended_crops[:5]):
                confidence = max(0.7, 0.95 - i * 0.05)
                crop_list.append({
                    "name": crop,
                    "confidence": round(confidence, 2),
                    "reason": f"Suitable for current soil and climate conditions"
                })
            
            return crop_list
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return self._get_fallback_crops_formatted()
    
    def _get_fallback_crops(self) -> List[str]:
        """Fallback crop recommendations"""
        return ['Rice', 'Cotton', 'Wheat', 'Onion', 'Tomato']
    
    def _get_fallback_crops_formatted(self) -> List[Dict]:
        """Formatted fallback recommendations"""
        crops = self._get_fallback_crops()
        return [
            {
                "name": crop,
                "confidence": max(0.7, 0.95 - i * 0.05),
                "reason": "Suitable for the region"
            }
            for i, crop in enumerate(crops)
        ]
