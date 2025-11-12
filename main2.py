import sqlite3
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import uvicorn
import os
import requests
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import asyncio
import re

from crop_model import CropRecommendationModel
from price_model import PricePredictionModel

load_dotenv()

DB_PATH = 'crops_data.db'
soil_data_df = None

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS crop_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_name TEXT NOT NULL,
            district TEXT NOT NULL,
            state TEXT NOT NULL,
            last_found DATE DEFAULT CURRENT_DATE
        )
    ''')
    conn.commit()
    conn.close()

def add_crop_location(crop_name, district, state):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO crop_locations (crop_name, district, state, last_found)
        VALUES (?, ?, ?, DATE('now'))
    ''', (crop_name, district, state))
    conn.commit()
    conn.close()

def load_soil_dataset():
    global soil_data_df
    try:
        soil_data_df = pd.read_csv('CropDataset-Enhanced.csv')
        print(f"✅ Loaded {len(soil_data_df)} soil data points from CSV")
    except Exception as e:
        print(f"❌ Failed to load soil dataset: {e}")
        soil_data_df = None

def safe_float(val):
    """Safely convert value to float"""
    try:
        return float(val)
    except:
        return 0.0

def predict_soil(lat: float, lng: float) -> dict:
    """Find nearest location in dataset and return its soil properties using KNN"""
    if soil_data_df is None:
        return {
            'N': 120, 'P': 40, 'K': 150, 'ph': 7.0,
            'N_level': 'Medium', 'P_level': 'Medium',
            'K_level': 'Medium', 'pH_level': 'Neutral'
        }
    
    try:
        coords = soil_data_df[['Latitude', 'Longitude']].values
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(coords)
        distances, indices = knn.kneighbors([[lat, lng]])
        nearest_idx = indices[0][0]
        nearest_row = soil_data_df.iloc[nearest_idx]
        
        # Convert to float safely
        n_high = safe_float(nearest_row.get('Nitrogen - High', 0))
        n_med = safe_float(nearest_row.get('Nitrogen - Medium', 0))
        n_low = safe_float(nearest_row.get('Nitrogen - Low', 0))
        
        if n_high > n_med and n_high > n_low:
            n_level = 'High'
        elif n_low > n_med:
            n_level = 'Low'
        else:
            n_level = 'Medium'
        
        p_high = safe_float(nearest_row.get('Phosphorous - High', 0))
        p_med = safe_float(nearest_row.get('Phosphorous - Medium', 0))
        p_low = safe_float(nearest_row.get('Phosphorous - Low', 0))
        
        if p_high > p_med and p_high > p_low:
            p_level = 'High'
        elif p_low > p_med:
            p_level = 'Low'
        else:
            p_level = 'Medium'
        
        k_high = safe_float(nearest_row.get('Potassium - High', 0))
        k_med = safe_float(nearest_row.get('Potassium - Medium', 0))
        k_low = safe_float(nearest_row.get('Potassium - Low', 0))
        
        if k_high > k_med and k_high > k_low:
            k_level = 'High'
        elif k_low > k_med:
            k_level = 'Low'
        else:
            k_level = 'Medium'
        
        ph_acidic = safe_float(nearest_row.get('pH - Acidic', 0))
        ph_neutral = safe_float(nearest_row.get('pH - Neutral', 0))
        ph_alkaline = safe_float(nearest_row.get('pH - Alkaline', 0))
        
        if ph_acidic > ph_neutral and ph_acidic > ph_alkaline:
            ph_level = 'Acidic'
        elif ph_alkaline > ph_neutral:
            ph_level = 'Alkaline'
        else:
            ph_level = 'Neutral'
        
        print(f"📍 Nearest soil: {nearest_row.get('Address', 'Unknown')}, Distance: {distances[0][0]:.4f}°")
        
        return {
            'N': int(n_high) if n_high > 0 else 120,
            'P': int(p_high) if p_high > 0 else 40,
            'K': int(k_high) if k_high > 0 else 150,
            'ph': 7.0 if ph_level == 'Neutral' else 6.0 if ph_level == 'Acidic' else 8.0,
            'N_level': n_level,
            'P_level': p_level,
            'K_level': k_level,
            'pH_level': ph_level
        }
        
    except Exception as e:
        print(f"❌ Soil prediction error: {e}")
        return {
            'N': 120, 'P': 40, 'K': 150, 'ph': 7.0,
            'N_level': 'Medium', 'P_level': 'Medium',
            'K_level': 'Medium', 'pH_level': 'Neutral'
        }

crop_model = None
price_model = None
tavily_client = None
gemini_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global crop_model, price_model, tavily_client, gemini_client
    print("🚀 Starting Smart Crop Recommendation System...")

    init_db()
    load_soil_dataset()

    crop_model = CropRecommendationModel()
    price_model = PricePredictionModel()

    try:
        from tavily import TavilyClient
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            tavily_client = TavilyClient(api_key=tavily_key)
            print("✅ Tavily Search API initialized")
        else:
            print("⚠️ TAVILY_API_KEY not found")
            tavily_client = None
    except Exception as e:
        print(f"⚠️ Tavily initialization failed: {e}")
        tavily_client = None

    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            gemini_client = genai.Client(api_key=api_key)
            print("✅ Gemini API initialized")
        else:
            print("⚠️ No Gemini API key found")
            gemini_client = None
    except Exception as e:
        print(f"⚠️ Gemini initialization failed: {e}")
        gemini_client = None

    print("✅ All systems ready!")
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="Smart Crop Recommendation System",
    description="AI-powered crop advisor for Indian farmers",
    version="6.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class RecommendationRequest(BaseModel):
    latitude: float
    longitude: float

class CropDetailsRequest(BaseModel):
    crop_name: str
    latitude: float
    longitude: float
    location_name: str

def get_location_info(latitude: float, longitude: float) -> Optional[Dict]:
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': latitude,
            'lon': longitude,
            'format': 'json',
            'addressdetails': 1
        }
        headers = {'User-Agent': 'CropRecommender/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        address = data.get('address', {})
        location = {
            'country': address.get('country'),
            'state': address.get('state'),
            'district': address.get('state_district') or address.get('county'),
            'subdistrict': address.get('suburb') or address.get('town'),
            'village': address.get('village'),
            'full_address': data.get('display_name')
        }
        if location['country'] != 'India':
            return None
        return location
    except Exception as e:
        print(f"❌ Location error: {e}")
        return None

def generate_price_chart(price_history: List[Dict], crop_name: str) -> str:
    try:
        dates = [p['date'] for p in price_history]
        prices_kg = [p['price'] / 100 for p in price_history]
        plt.figure(figsize=(10, 5))
        plt.plot(dates, prices_kg, marker='o', linewidth=2, markersize=8, 
                color='#4caf50', markerfacecolor='#4caf50', markeredgecolor='white', markeredgewidth=2)
        plt.fill_between(range(len(dates)), prices_kg, alpha=0.2, color='#4caf50')
        plt.title(f'{crop_name} - Price Trend & Prediction', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Month', fontsize=11, fontweight='bold')
        plt.ylabel('Price per Kg (₹)', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if len(prices_kg) >= 2:
            plt.axvline(x=len(dates)-2, color='blue', linestyle='--', alpha=0.5, label='Today')
            plt.axvline(x=len(dates)-1, color='orange', linestyle='--', alpha=0.5, label='Future Price')
            plt.legend()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"❌ Chart generation error: {e}")
        return ""

def search_local_crops_tavily(district: str, state: str) -> List[str]:
    if not tavily_client:
        return []
    try:
        query = f"crops grown in {district} district {state} India 2025"
        print(f"🔍 Tavily searching: {query}")
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=10,
            include_domains=[
                "agmarknet.gov.in",
                "enam.gov.in",
                "agricoop.nic.in",
                "krishi.maharashtra.gov.in",
                "indianexpress.com"
            ]
        )
        common_crops = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Soybean', 
                        'Maize', 'Jowar', 'Bajra', 'Groundnut', 'Onion', 
                        'Tomato', 'Potato', 'Chilli', 'Turmeric', 'Pulses']
        found_crops = []
        if 'answer' in response and response['answer']:
            text = response['answer'].lower()
            for crop in common_crops:
                if crop.lower() in text:
                    found_crops.append(crop)
        if 'results' in response:
            for result in response['results']:
                content = (result.get('content', '') + ' ' + result.get('title', '')).lower()
                for crop in common_crops:
                    if crop.lower() in content and crop not in found_crops:
                        found_crops.append(crop)
        print(f"✅ Tavily found crops: {found_crops[:10]}")
        return found_crops[:10]
    except Exception as e:
        print(f"⚠️ Tavily search error: {e}")
        return []

def extract_price_from_web_results(web_results, district, state, crop_name):
    price_pattern = r"(?:\b|₹)(\d{3,5})\s*(?:per\s*quintal|/quintal|quintal|Rs|रुपये)"
    fallback_price = None
    fallback_source = None
    for result in web_results:
        text_combo = (result.get("content", "") + " " + result.get("title", ""))
        if (district.lower() in text_combo.lower() or state.lower() in text_combo.lower()) and crop_name.lower() in text_combo.lower():
            matches = re.findall(price_pattern, text_combo)
            if matches:
                try:
                    price = int(matches[0])
                    return price, result.get('url', '')
                except:
                    continue
        if (state.lower() in text_combo.lower() or crop_name.lower() in text_combo.lower()) and not fallback_price:
            matches = re.findall(price_pattern, text_combo)
            if matches:
                try:
                    fallback_price = int(matches[0])
                    fallback_source = result.get('url', '')
                except:
                    continue
    if fallback_price:
        return fallback_price, fallback_source
    return None, None

def get_crop_info_tavily(crop_name: str, district: str, state: str) -> Dict:
    if not tavily_client:
        return {"success": False, "content": "", "sources": [], "results": []}
    try:
        queries = [
            f"{crop_name} mandi price {district} {state} October 2025",
            f"{crop_name} cultivation guide {state} India fertilizers",
            f"{crop_name} farming precautions India diseases pests"
        ]
        all_results = []
        combined_content = []
        for query in queries:
            print(f"🔍 Searching: {query}")
            try:
                response = tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=3
                )
                if 'results' in response:
                    all_results.extend(response['results'])
                if 'answer' in response and response['answer']:
                    combined_content.append(response['answer'])
            except Exception as e:
                print(f"⚠️ Query failed: {e}")
                continue
        print(f"✅ Found {len(all_results)} total sources")
        return {
            "success": len(all_results) > 0,
            "content": "\n\n".join(combined_content),
            "sources": all_results[:8],
            "results": all_results
        }
    except Exception as e:
        print(f"❌ Tavily error: {e}")
        return {"success": False, "content": "", "sources": [], "results": []}

def generate_search_prompt(crop_name: str, district: str, state: str) -> str:
    trusted_urls = [
        f"https://agmarknet.gov.in",
        f"https://enam.gov.in",
        f"http://krishi.maharashtra.gov.in",
        f"https://agricoop.nic.in",
        f"https://www.indianexpress.com",
        "https://www.google.com"
    ]
    url_list_str = "\n".join([f"- {url}" for url in trusted_urls])

    prompt = f"""
You are an expert agricultural researcher.

Search the following trusted sources:
{url_list_str}

Your task: Based on current and real-time data, provide detailed information about growing {crop_name} in {district} district, {state} state, INDIA in 2025.

Provide:
1. Current market price (₹ per quintal/kg)
2. Harvesting timeline and growing duration
3. Fertilizer names recommended (no quantities)
4. Common precautions during growing periods
5. Estimated revenue based on market price
6. Recent government schemes or subsidies

Be concise, detailed, and practical for farmers.
"""
    return prompt

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.post("/recommend")
async def recommend(req: RecommendationRequest):
    try:
        print(f"\n{'='*60}")
        print(f"📍 Request: {req.latitude}, {req.longitude}")

        location = get_location_info(req.latitude, req.longitude)
        if not location or location['country'] != 'India':
            raise HTTPException(400, "Please select a location in India")
        print(f"✅ Location: {location['district']}, {location['state']}")

        soil = predict_soil(req.latitude, req.longitude)

        web_crops = search_local_crops_tavily(location['district'], location['state'])
        print(f"🌐 Tavily found: {web_crops}")

        ml_crops = crop_model.predict(req.latitude, req.longitude, soil)
        ml_crop_names = [c['name'] for c in ml_crops[:5]]
        print(f"🤖 ML predicted: {ml_crop_names}")

        for crop_name in web_crops:
            if crop_name not in ml_crop_names:
                add_crop_location(crop_name, location['district'], location['state'])

        final_crops = web_crops if web_crops else ml_crop_names

        # Just return crop names with basic price from model - NO detailed web search here
        crops_with_prices = []
        for crop_name in final_crops:
            try:
                price_pred = price_model.predict(
                    crop_name,
                    {'state': location['state'], 'district': location['district']},
                    soil,
                    None
                )
                crops_with_prices.append({
                    'name': crop_name,
                    'source': 'Web-verified' if crop_name in web_crops else 'AI-predicted',
                    'current_price': price_pred['predicted_price_per_quintal'],
                    'price_range': price_pred['price_range'],
                    'confidence': 0.85 if crop_name in web_crops else 0.7,
                })
            except Exception as e:
                print(f"⚠️ Price error for {crop_name}: {e}")
                crops_with_prices.append({
                    'name': crop_name,
                    'source': 'Web-verified' if crop_name in web_crops else 'AI-predicted',
                    'current_price': 2000,
                    'price_range': {'min': 1800, 'max': 2200},
                    'confidence': 0.7,
                })

        print(f"✅ Returning {len(crops_with_prices)} crops\n")

        return {
            'crops': crops_with_prices,
            'location': location,
            'soil': soil,
            'timestamp': datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.post("/crop-details")
async def crop_details(req: CropDetailsRequest):
    try:
        print(f"\n{'='*60}")
        print(f"📋 Details for: {req.crop_name} at {req.location_name}")

        parts = req.location_name.split(',')
        district = parts[0].strip() if len(parts) > 0 else "Unknown"
        state = parts[1].strip() if len(parts) > 1 else "Maharashtra"

        soil = predict_soil(req.latitude, req.longitude)
        price_pred = price_model.predict(
            req.crop_name,
            {'state': state, 'district': district},
            soil,
            None
        )

        current_price = price_pred['predicted_price_per_quintal']

        growth_days = {
            'Rice': 120, 'Wheat': 120, 'Cotton': 150, 'Onion': 120, 
            'Tomato': 90, 'Potato': 90, 'Sugarcane': 365, 'Soybean': 100,
            'Maize': 90, 'Groundnut': 120, 'Bajra': 75, 'Jowar': 100,
            'Chilli': 150, 'Turmeric': 270, 'Pulses': 90
        }
        days_to_harvest = growth_days.get(req.crop_name, 120)
        planting_date = datetime.now()
        harvest_date = planting_date + timedelta(days=days_to_harvest)
        selling_date = harvest_date + timedelta(days=7)

        future_price_pred = price_model.predict(
            req.crop_name,
            {'state': state, 'district': district},
            soil,
            selling_date.strftime('%Y-%m-%d')
        )
        future_price = future_price_pred['predicted_price_per_quintal']

        # Generate price history - ONLY LAST 2 MONTHS + TODAY + FUTURE
        import random
        price_history = []
        
        # Last 2 months (2 data points)
        for i in range(-2, 0):
            month_date = datetime.now() + timedelta(days=i*30)
            # Generate realistic variation around current price
            price_var = current_price * (0.90 + random.random() * 0.10)  # 90-100% of current
            price_history.append({
                'date': month_date.strftime('%b %Y'),
                'price': int(price_var)
            })
        
        # TODAY - use exact current price
        price_history.append({
            'date': datetime.now().strftime('%b %Y'),
            'price': current_price
        })
        
        # FUTURE - use exact predicted future price
        price_history.append({
            'date': selling_date.strftime('%b %Y'),
            'price': future_price
        })

        chart_base64 = generate_price_chart(price_history, req.crop_name)

        tavily_data = get_crop_info_tavily(req.crop_name, district, state)

        if gemini_client and tavily_data['success']:
            try:
                synthesis_prompt = generate_search_prompt(req.crop_name, district, state)

                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=synthesis_prompt,
                    config={
                        "temperature": 0.3,
                        "top_p": 0.95,
                        "top_k": 40
                    }
                )
                detailed_info = response.text
                print("✅ Gemini synthesis successful")
            except Exception as e:
                print(f"⚠️ Gemini synthesis failed: {e}")
                detailed_info = tavily_data['content'] or f"**{req.crop_name} Farming Information**\n\nBased on web sources, {req.crop_name} is grown in {district}. Please consult local agricultural extension officers for detailed guidance."
        else:
            detailed_info = f"**{req.crop_name} Farming Guide**\n\nDetailed information is being gathered. Please consult with local agricultural extension officers for specific guidance on growing {req.crop_name} in {district}.\n\n**Contact**: Kisan Call Center - 1800-180-1551"

        print(f"✅ Generated details with {len(tavily_data.get('sources', []))} sources\n")

        return {
            'crop_name': req.crop_name,
            'location': {'district': district, 'state': state},
            'pricing': {
                'current_price_per_quintal': current_price,
                'future_price_per_quintal': future_price,
                'price_per_kg': current_price / 100,
                'future_price_per_kg': future_price / 100,
                'price_history': price_history
            },
            'timeline': {
                'planting_date': planting_date.strftime('%d %b %Y'),
                'harvest_date': harvest_date.strftime('%d %b %Y'),
                'selling_date': selling_date.strftime('%d %b %Y'),
                'days_to_harvest': days_to_harvest
            },
            'chart_image': chart_base64,
            'detailed_info': detailed_info,
            'sources': [
                {
                    'title': s.get('title', 'Web Source'),
                    'url': s.get('url', '')
                }
                for s in tavily_data.get('sources', [])[:5]
            ],
            'generated_at': datetime.now().strftime('%d %B %Y, %I:%M %p')
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))




# ==================== WEBSOCKET INTEGRATION ====================

class ConnectionManager:
    """Manages WebSocket connections"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Send error: {e}")

manager = ConnectionManager()

async def perform_search_with_updates(crop_name: str, district: str, state: str, websocket: WebSocket):
    """Perform search with real-time progress updates"""
    try:
        print(f"🔍 Search: {crop_name} in {district}, {state}")

        # Update 1: Starting (10%)
        await manager.send_message({
            "type": "browser_status",
            "status": f"🚀 Initializing search for {crop_name}...",
            "progress": 10
        }, websocket)
        await asyncio.sleep(0.5)

        # Update 2: Searching (30%)
        await manager.send_message({
            "type": "browser_status",
            "status": f"🔍 Searching {crop_name} in {district}, {state}...",
            "progress": 30
        }, websocket)
        await asyncio.sleep(0.5)

        # Actual search
        results = []
        if tavily_client:
            query = f"{crop_name} crop cultivation price {district} {state} India 2025"
            print(f"📡 Query: {query}")

            # Update 3: Querying (50%)
            await manager.send_message({
                "type": "browser_status",
                "status": "📡 Querying databases...",
                "progress": 50
            }, websocket)

            response = tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )
            results = response.get('results', [])
            print(f"✅ Found {len(results)} results")

            # Update 4: Processing (70%)
            await manager.send_message({
                "type": "browser_status",
                "status": f"📊 Processing {len(results)} sources...",
                "progress": 70
            }, websocket)
            await asyncio.sleep(0.5)

            # Update 5: Complete (100%)
            await manager.send_message({
                "type": "browser_status",
                "status": "✅ Search complete!",
                "progress": 100
            }, websocket)

            # Format results
            formatted = []
            for r in results[:5]:
                formatted.append({
                    "title": r.get('title', 'No title'),
                    "url": r.get('url', ''),
                    "content": r.get('content', '')[:300] + '...'
                })

            # Send results
            await manager.send_message({
                "type": "search_complete",
                "results_count": len(results),
                "results": formatted
            }, websocket)

        else:
            await manager.send_message({
                "type": "error",
                "message": "Search not available"
            }, websocket)

    except Exception as e:
        print(f"❌ Error: {e}")
        await manager.send_message({
            "type": "error",
            "message": str(e)
        }, websocket)

@app.websocket("/ws/browser-stream")
async def websocket_browser_stream(websocket: WebSocket):
    print("🔵 WebSocket connection")
    try:
        await websocket.accept()
        print("✅ Accepted")

        await manager.connect(websocket)
        await websocket.send_json({"type": "connection_established", "status": "Connected"})

        while True:
            data = await websocket.receive_json()
            print(f"📨 {data.get('type')}")

            if data.get("type") == "start_search":
                crop = data.get("crop_name", "Rice")
                district = data.get("district", "Unknown")
                state = data.get("state", "Unknown")

                await perform_search_with_updates(crop, district, state, websocket)

    except WebSocketDisconnect:
        print("🔌 Disconnected")
    except Exception as e:
        print(f"❌ {e}")
        traceback.print_exc()
    finally:
        manager.disconnect(websocket)

# ==================== END WEBSOCKET ====================


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
