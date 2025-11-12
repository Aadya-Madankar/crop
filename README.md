# 🌾 Smart Crop Recommendation System
## AI-Powered Crop Advisory for Indian Farmers

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Features](#features)
5. [Data Flow](#data-flow)
6. [API Endpoints](#api-endpoints)
7. [Machine Learning Models](#machine-learning-models)
8. [Installation Guide](#installation-guide)
9. [Usage Guide](#usage-guide)
10. [Database Schema](#database-schema)
11. [Future Enhancements](#future-enhancements)

---

## 1. Project Overview

### 1.1 Purpose
The Smart Crop Recommendation System is an AI-powered web application designed to help Indian farmers make informed decisions about crop selection based on:
- Geographic location (latitude/longitude)
- Soil characteristics (N, P, K, pH levels)
- Real-time market prices
- Historical data analysis
- Web-verified local crop information

### 1.2 Target Users
- Indian farmers seeking crop recommendations
- Agricultural consultants
- Government agricultural departments
- Agricultural students and researchers

### 1.3 Key Benefits
- **Data-Driven Decisions**: Combines ML predictions with real-world web data
- **Price Predictions**: Forecasts future crop prices to maximize profit
- **Location-Specific**: Uses KNN algorithm to find nearest soil data
- **Market Intelligence**: Searches live web data for current crop trends
- **Revenue Calculator**: Interactive tool to estimate earnings

---

## 2. System Architecture

### 2.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│  (HTML/CSS/JavaScript + Leaflet Maps + Interactive Charts) │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  /recommend          │  /crop-details                │  │
│  │  - Location input    │  - Detailed analysis          │  │
│  │  - Crop suggestions  │  - Price predictions          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────┬──────────────┬──────────────┬─────────────┬──────────┘
      │              │              │             │
      ▼              ▼              ▼             ▼
┌──────────┐  ┌───────────┐  ┌──────────┐  ┌─────────┐
│   ML     │  │   Soil    │  │  Tavily  │  │ Gemini  │
│  Models  │  │  Dataset  │  │  Search  │  │   AI    │
│          │  │   (KNN)   │  │   API    │  │   API   │
└──────────┘  └───────────┘  └──────────┘  └─────────┘
      │              │              │             │
      └──────────────┴──────────────┴─────────────┘
                     │
                     ▼
            ┌────────────────┐
            │   SQLite DB    │
            │  (crop_data)   │
            └────────────────┘
```

### 2.2 Component Description

**Frontend Layer:**
- Interactive map (Leaflet.js) for location selection
- Responsive crop cards with price display
- Modal dialogs for detailed crop information
- Revenue calculator with sliders
- Price trend charts (matplotlib)

**Backend Layer (FastAPI):**
- `/recommend` - Main recommendation endpoint
- `/crop-details` - Detailed crop analysis endpoint
- Location geocoding (Nominatim API)
- Real-time web search integration

**AI/ML Layer:**
- Crop Recommendation Model (Multi-label classification)
- Price Prediction Model (Ensemble ML)
- KNN for soil data matching

**Data Layer:**
- SQLite database for crop locations
- CSV dataset with 400+ soil samples across India
- Historical price data

---

## 3. Technology Stack

### 3.1 Backend Technologies
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Core programming language |
| FastAPI | Latest | REST API framework |
| Uvicorn | Latest | ASGI server |
| SQLite3 | Built-in | Database |
| Pandas | Latest | Data processing |
| NumPy | Latest | Numerical computations |
| Scikit-learn | Latest | Machine learning |
| Matplotlib | Latest | Chart generation |

### 3.2 Frontend Technologies
| Technology | Purpose |
|------------|---------|
| HTML5/CSS3 | Structure and styling |
| JavaScript (ES6+) | Client-side logic |
| Leaflet.js | Interactive maps |
| Fetch API | HTTP requests |

### 3.3 External APIs
| API | Purpose |
|-----|---------|
| Nominatim (OSM) | Reverse geocoding |
| Tavily Search | Web search for crops |
| Google Gemini | AI synthesis |

### 3.4 Key Python Libraries
```
fastapi
uvicorn
pandas
numpy
scikit-learn
matplotlib
python-dotenv
requests
joblib
```

---

## 4. Features

### 4.1 Core Features

#### 🗺️ Interactive Map Selection
- Click anywhere in India to get recommendations
- Visual marker placement
- Automatic location detection
- District and state identification

#### 🌱 Dual-Source Recommendations
- **ML-Predicted Crops**: Based on trained models
- **Web-Verified Crops**: Real-time search results
- **Hybrid Approach**: Combines both for accuracy

#### 💰 Real-Time Price Display
- Current market price per kg/quintal
- Price range (min-max)
- Confidence level indicator
- Source badge (ML/Web/Both)

#### 📊 Price Trend Analysis
- Historical price charts (last 2 months)
- Future price predictions
- Visual trend lines
- Interactive legends

#### 🧮 Revenue Calculator
- Adjustable quantity slider (1-100 kg)
- Today's revenue calculation
- Future revenue projection
- Profit/loss analysis
- Visual indicators (green/red)

#### 📋 Comprehensive Growing Guide
- Cultivation practices
- Fertilizer recommendations
- Pest and disease precautions
- Timeline information
- Government schemes

### 4.2 Technical Features

#### Soil Analysis (KNN-Based)
- Finds nearest soil data point
- Returns N, P, K, pH levels
- Categorizes as High/Medium/Low
- pH classification (Acidic/Neutral/Alkaline)

#### Smart Search Integration
- Searches agricultural databases
- Prioritizes trusted sources (agmarknet.gov.in, etc.)
- Extracts local crop information
- Price extraction from web results

#### AI Synthesis (Gemini)
- Combines multiple sources
- Generates comprehensive guides
- Structured information output
- Context-aware responses

---

## 5. Data Flow

### 5.1 Recommendation Flow

```
User clicks map location
         │
         ▼
Extract lat/lng
         │
         ▼
Reverse geocode (Nominatim)
         │
         ├─────────────┐
         ▼             ▼
   Get soil data   Search web
   (KNN nearest)   (Tavily API)
         │             │
         ▼             ▼
   ML prediction   Extract crops
         │             │
         └──────┬──────┘
                ▼
         Merge results
                │
                ▼
         Get prices (ML model)
                │
                ▼
       Return ranked crops
                │
                ▼
         Display to user
```

### 5.2 Detailed Crop Analysis Flow

```
User clicks crop card
         │
         ▼
Get crop name + location
         │
         ├──────────────────┐
         ▼                  ▼
  Predict prices     Search web data
  (Current + Future) (Tavily API)
         │                  │
         ▼                  ▼
  Generate chart     Extract info
  (Matplotlib)       (Market prices)
         │                  │
         └────────┬─────────┘
                  ▼
          Synthesize with AI
          (Gemini API)
                  │
                  ▼
          Format response
                  │
                  ▼
          Display in modal
```

### 5.3 Price Prediction Flow

```
Crop + Location + Soil + Date
         │
         ▼
Load trained models
         │
         ├─────────────────────┐
         ▼                     ▼
  ML Model Available?    Use MSP Fallback
  (Random Forest,        (Base prices)
   Gradient Boost,
   Linear Regression)
         │                     │
         ▼                     │
  Ensemble prediction          │
         │                     │
         ▼                     │
  Apply soil multiplier        │
  (Quality adjustment)         │
         │                     │
         └──────────┬──────────┘
                    ▼
          Return price + range
```

---

## 6. API Endpoints

### 6.1 GET `/`
**Description**: Serves the main HTML interface

**Response**: HTML page

---

### 6.2 POST `/recommend`

**Description**: Get crop recommendations for a location

**Request Body**:
```json
{
  "latitude": 18.5204,
  "longitude": 73.8567
}
```

**Response**:
```json
{
  "crops": [
    {
      "name": "Rice",
      "source": "Web-verified",
      "current_price": 2320,
      "price_range": {
        "min": 1972,
        "max": 2668
      },
      "confidence": 0.85
    }
  ],
  "location": {
    "country": "India",
    "state": "Maharashtra",
    "district": "Pune",
    "full_address": "Pune, Maharashtra, India"
  },
  "soil": {
    "N": 120,
    "P": 40,
    "K": 150,
    "ph": 7.0,
    "N_level": "Medium",
    "P_level": "Medium",
    "K_level": "Medium",
    "pH_level": "Neutral"
  },
  "timestamp": "2025-11-12T10:30:00"
}
```

**Process**:
1. Validate location is in India
2. Get soil data using KNN
3. Search web for local crops
4. Get ML predictions
5. Merge and rank results
6. Fetch current prices
7. Return combined response

---

### 6.3 POST `/crop-details`

**Description**: Get detailed analysis for a specific crop

**Request Body**:
```json
{
  "crop_name": "Rice",
  "latitude": 18.5204,
  "longitude": 73.8567,
  "location_name": "Pune, Maharashtra"
}
```

**Response**:
```json
{
  "crop_name": "Rice",
  "location": {
    "district": "Pune",
    "state": "Maharashtra"
  },
  "pricing": {
    "current_price_per_quintal": 2320,
    "future_price_per_quintal": 2450,
    "price_per_kg": 23.20,
    "future_price_per_kg": 24.50,
    "price_history": [
      {"date": "Sep 2025", "price": 2200},
      {"date": "Oct 2025", "price": 2280},
      {"date": "Nov 2025", "price": 2320},
      {"date": "Feb 2026", "price": 2450}
    ]
  },
  "timeline": {
    "planting_date": "12 Nov 2025",
    "harvest_date": "12 Mar 2026",
    "selling_date": "19 Mar 2026",
    "days_to_harvest": 120
  },
  "chart_image": "data:image/png;base64,...",
  "detailed_info": "## Rice Cultivation Guide\n\n### Current Market...",
  "sources": [
    {
      "title": "Agmarknet - Rice Prices",
      "url": "https://agmarknet.gov.in/..."
    }
  ],
  "generated_at": "12 November 2025, 10:30 AM"
}
```

---

## 7. Machine Learning Models

### 7.1 Crop Recommendation Model

**Type**: Multi-label Classification

**Input Features**:
- Latitude
- Longitude
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- pH level

**Output**: List of suitable crops

**File**: `models/crop_recommender_model.pkl`

**Fallback Crops**: 
`['Rice', 'Cotton', 'Wheat', 'Onion', 'Tomato']`

---

### 7.2 Price Prediction Model

**Type**: Ensemble Model (Random Forest + Gradient Boosting + Linear Regression)

**Input Features**:
- State (encoded)
- District (encoded)
- Commodity (encoded)
- Year, Month, Day
- Day of Week
- Quarter
- Min/Max historical prices
- Price range and volatility

**Output**: 
- Predicted price per quintal
- Price range (min/max)
- Confidence level

**Weights**:
- Random Forest: 50%
- Gradient Boosting: 30%
- Linear Regression: 20%

**Soil Quality Adjustment**:
- High nutrients: +5% to +3%
- Low nutrients: -3% to -2%
- Neutral pH: +2%
- Total range: 0.85x to 1.25x

**MSP Fallback Prices** (2025):
```python
{
    'Rice': 2320, 'Wheat': 2425, 'Cotton': 6969,
    'Tomato': 2200, 'Onion': 1800, 'Potato': 1400,
    'Maize': 2090, 'Soybean': 4600, 'Sugarcane': 340,
    'Groundnut': 6377, 'Bajra': 2625, 'Jowar': 3570
}
```

---

### 7.3 KNN Soil Matching

**Algorithm**: K-Nearest Neighbors (k=1)

**Distance Metric**: Euclidean

**Dataset**: 400+ soil samples from CropDataset-Enhanced.csv

**Process**:
1. Input: User's latitude/longitude
2. Find nearest soil sample in dataset
3. Extract N, P, K, pH levels
4. Categorize levels (High/Medium/Low)
5. Return soil characteristics

---

## 8. Installation Guide

### 8.1 Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for APIs)

### 8.2 Step-by-Step Installation

**Step 1: Clone/Download Project**
```bash
# If using git
git clone <repository-url>
cd smart-crop-advisor

# Or extract downloaded ZIP
```

**Step 2: Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**Step 3: Install Dependencies**
```bash
pip install fastapi uvicorn pandas numpy scikit-learn matplotlib python-dotenv requests joblib tavily-python google-generativeai
```

**Step 4: Setup Environment Variables**

Create `.env` file in root directory:
```env
GEMINI_API_KEY=your_gemini_key_here
TAVILY_API_KEY=your_tavily_key_here
```

**Step 5: Verify Files**

Ensure these files exist:
```
├── main2.py
├── crop_model.py
├── price_model.py
├── index.html
├── CropDataset-Enhanced.csv
├── models/
│   ├── crop_recommender_model.pkl
│   └── crop_price_prediction_model.pkl
├── .env
└── crops_data.db (will be created automatically)
```

**Step 6: Run Application**
```bash
python main2.py
```

**Step 7: Access Application**

Open browser and navigate to:
```
http://localhost:8000
```

---

## 9. Usage Guide

### 9.1 Getting Crop Recommendations

**Step 1**: Open application in browser

**Step 2**: Click anywhere on the map within India

**Step 3**: Wait for system to:
- Identify location
- Search for local crops
- Run ML predictions
- Fetch current prices

**Step 4**: View results in crop cards showing:
- Crop name with emoji
- Source badge (ML/Web/Both)
- Confidence percentage
- Current market price
- Price range

### 9.2 Viewing Detailed Analysis

**Step 1**: Click on any crop card

**Step 2**: Modal opens showing:
- Current and future prices
- Days to harvest
- Selling date recommendation
- Price trend chart

**Step 3**: Use revenue calculator:
- Adjust quantity slider
- See today's revenue
- See future revenue
- View profit/loss

**Step 4**: Scroll down for:
- Complete growing guide
- Fertilizer recommendations
- Pest control measures
- Government schemes
- Information sources

### 9.3 Understanding Results

**Crop Badges**:
- 🤖 **AI Predicted**: From ML model only
- 🌐 **Web Found**: Verified from web search
- ✅ **ML + Web**: Both sources agree
- **X% Match**: Confidence level

**Price Colors**:
- 🟢 **Green**: Profit expected
- 🔴 **Red**: Loss expected
- 🟠 **Orange**: Current prices

---

## 10. Database Schema

### 10.1 SQLite Database

**Database Name**: `crops_data.db`

**Table**: `crop_locations`

```sql
CREATE TABLE crop_locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crop_name TEXT NOT NULL,
    district TEXT NOT NULL,
    state TEXT NOT NULL,
    last_found DATE DEFAULT CURRENT_DATE
);
```

**Purpose**: Track crops found via web search for future reference

**Operations**:
- INSERT: When web search finds a new crop-location pair
- SELECT: Could be used for analytics (not currently implemented)

---

### 10.2 CSV Dataset Structure

**File**: `CropDataset-Enhanced.csv`

**Columns**:
- Address, Status geocode, Formatted address
- Latitude, Longitude
- Type, Location Type
- Country, Region
- Crop (comma-separated list)
- Nitrogen - High/Medium/Low (%)
- Phosphorous - High/Medium/Low (%)
- Potassium - High/Medium/Low (%)
- pH - Acidic/Neutral/Alkaline (%)

**Sample Row**:
```
Pune, Maharashtra, India | 18.5204 | 73.8567 | Sugarcane, Jowar... | N-Low: 89.41% | P-High: 24.12% | K-High: 67.81% | pH-Neutral: 98.95%
```

---

## 11. Future Enhancements

### 11.1 Planned Features
- 📱 Mobile app (Android/iOS)
- 🌦️ Weather integration
- 📈 Historical trend analysis
- 👥 User accounts and saved locations
- 📊 Dashboard with analytics
- 🗣️ Multi-language support (Hindi, Marathi, etc.)
- 📱 SMS/WhatsApp notifications
- 🎯 Personalized recommendations based on farm size

### 11.2 Technical Improvements
- Real-time price updates from mandis
- More ML model types (Deep Learning)
- Blockchain for price transparency
- IoT sensor integration
- Drone imagery analysis
- Crop disease detection (Computer Vision)

### 11.3 Data Enhancements
- More soil samples (1000+ locations)
- Real mandi integration
- Satellite imagery data
- Climate change predictions
- Water availability data

---

## 📞 Support & Contact

For queries or issues:
- Check error logs in console
- Verify API keys in .env file
- Ensure all dependencies are installed
- Check network connectivity for APIs

---

## 📄 License & Credits

**Project**: Smart Crop Recommendation System  
**Purpose**: Educational/Research  
**APIs Used**: 
- Nominatim (OpenStreetMap)
- Tavily Search API
- Google Gemini AI
- Leaflet Maps

**Disclaimer**: Price predictions are estimates based on historical data and ML models. Always verify with local agricultural markets and experts before making farming decisions.

---

## 🎓 Conclusion

This Smart Crop Recommendation System demonstrates the power of combining:
- Machine Learning (Scikit-learn models)
- Web Search Integration (Real-time data)
- Geographic Analysis (KNN for soil matching)
- AI Synthesis (Gemini for comprehensive guides)

The system provides farmers with data-driven insights while maintaining simplicity and accessibility through an intuitive web interface.

**Key Achievement**: Successfully bridges the gap between AI predictions and real-world agricultural data to help farmers make profitable crop decisions.

---


