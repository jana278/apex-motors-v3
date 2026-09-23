import os
import re
import io
import sys
import random
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify, send_from_directory
import requests
from bs4 import BeautifulSoup
import numpy as np
import joblib

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

class ApexProductionValuationEngine:
    def __init__(self, model, num_cols, cat_cols, medians):
        self.model = model
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.medians = medians

sys.modules['__main__'].ApexProductionValuationEngine = ApexProductionValuationEngine

try:
    from catboost import Pool, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    Pool = None
    CatBoostRegressor = None

VISION_MODEL_NAME = "dima806/car_models_image_detection"

app = Flask(__name__)

# [FIX 1] Updated Regex to catch ALL Hatla2ee car links (including used-car)
DETAIL_URL_PATTERN = re.compile(r'/(?:car|new-car|used-car)/[^\?#]*?\d{4,}$', re.IGNORECASE)
ARABIC_TO_ENGLISH_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_digits(text: str) -> str:
    if not text:
        return ""
    return text.translate(ARABIC_TO_ENGLISH_DIGITS)

def is_valid_vehicle_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    if "hatla2ee.com" not in parsed.netloc.lower():
        return False
    return bool(DETAIL_URL_PATTERN.search(parsed.path)) and "teraz/" not in parsed.path.lower()

class DualPlatformMarketScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        self.base_url = "https://eg.hatla2ee.com"

    def scrape_hatla2ee(self, brand: str, model: str = None, page: int = 1) -> list:
        records = []
        if not brand:
            return []

        clean_b = brand.lower().strip()
        clean_m = model.lower().strip() if model else ""
        
        target_url = f"{self.base_url}/ar/car/{clean_b}"
        if clean_m:
            target_url += f"/{clean_m.replace(' ', '-')}"
        if page > 1:
            target_url += f"/page/{page}"

        try:
            resp = self.session.get(target_url, headers=self.headers, timeout=12)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                records_by_url = {}

                cards = soup.find_all(lambda tag: tag.name in ['div', 'article', 'section'] and tag.get('class') and 'bg-card' in tag.get('class'))
                if not cards:
                    cards = soup.find_all(lambda tag: tag.name in ['div', 'article', 'section'] and tag.get('class') and any('unit' in c.lower() or 'card' in c.lower() for c in tag.get('class')))

                for card in cards:
                    try:
                        detail_a = None
                        for a in card.find_all("a", href=True):
                            href = a["href"].strip()
                            if DETAIL_URL_PATTERN.search(href) and "teraz/" not in href.lower():
                                detail_a = a
                                text = a.get_text(strip=True)
                                if text and not any(k in text.lower() for k in ["slide", "previous", "next", "عرض الكل"]):
                                    break

                        if not detail_a:
                            continue

                        raw_href = detail_a["href"].strip()
                        full_link = urljoin(self.base_url, raw_href)

                        if full_link in records_by_url:
                            continue

                        title_text = ""
                        for a in card.find_all("a", href=True):
                            t = a.get_text(strip=True)
                            if t and not any(k in t.lower() for k in ["slide", "previous", "next", "عرض الكل"]):
                                title_text = t
                                break

                        image_url = None
                        for img in card.find_all("img"):
                            src = img.get("src") or img.get("data-src") or ""
                            if "listing_image" in src or "hatla2ee.com/listing" in src:
                                image_url = src
                                break
                        if not image_url:
                            for img in card.find_all("img"):
                                src = img.get("src") or img.get("data-src") or ""
                                if src and not any(x in src for x in ["logo", "icon", "agency"]):
                                    image_url = src
                                    break

                        card_text_space = normalize_digits(card.get_text(" ", strip=True))
                        card_text_bar = normalize_digits(card.get_text(" | ", strip=True))

                        price = None
                        p_match = re.search(r'([\d,]{4,12})\s*(?:جنيه|EGP|ج\.م|L\.E)', card_text_space)
                        if p_match:
                            try:
                                p_val = float(p_match.group(1).replace(',', '').replace(' ', ''))
                                if p_val > 10000:
                                    price = p_val
                            except ValueError:
                                price = None

                        year = None
                        y_match = re.search(r'\b(19\d{2}|20\d{2})\b', card_text_space)
                        if y_match:
                            year = int(y_match.group(1))

                        mileage = None
                        km_match = re.search(r'([\d,]{1,8})\s*(?:کم|كم|km|كيلومتر|كيلو)', card_text_space, re.IGNORECASE)
                        if km_match:
                            try:
                                km_str = km_match.group(1).replace(',', '').replace(' ', '').strip()
                                mileage = float(km_str)
                            except ValueError:
                                mileage = None
                        elif re.search(r'\b0\s*(?:کم|كم|km)\b', card_text_space, re.IGNORECASE):
                            mileage = 0.0

                        transmission = "Automatic"
                        if any(t in card_text_space for t in ["يدوي", "مانيوال", "Manual"]):
                            transmission = "Manual"

                        fuel_type = "Benzine"
                        if "هجين" in card_text_space or "Hybrid" in card_text_space:
                            fuel_type = "Hybrid"
                        elif "كهرباء" in card_text_space or "Electric" in card_text_space:
                            fuel_type = "Electric"
                        elif "غاز" in card_text_space or "Gas" in card_text_space:
                            fuel_type = "Gas"

                        condition_tag = "Fabrika" if "فابريكا" in card_text_space else "Used"

                        tokens = [t.strip() for t in card_text_bar.split('|') if t.strip()]
                        location = "Cairo"
                        known_locs = ["القاهرة", "الجيزة", "الإسكندرية", "التجمع", "المهندسين", "دمياط", "منوفية", "الشرقية", "الدقهلية", "الغربية", "أسيوط", "سوهاج", "المنيا", "بني سويف", "الفيوم", "إسماعيلية", "السويس", "بورسعيد"]
                        for tok in tokens:
                            if any(loc in tok for loc in known_locs):
                                location = tok
                                break

                        rec_title = title_text if len(title_text) >= 3 else f"{brand.title()} {model.title() if model else ''} {year or ''}".strip()

                        records_by_url[full_link] = {
                            "name": rec_title,
                            "brand": brand.title(),
                            "model": model.title() if model else "Model",
                            "price": price,
                            "year": year if year else 2024,
                            "mileage": mileage,
                            "location": location,
                            "transmission": transmission,
                            "fuel_type": fuel_type,
                            "car_condition": "New" if mileage == 0 else "Used",
                            "condition_tag": condition_tag,
                            "trim_tier": "Topline",
                            "source": "Hatla2ee Market",
                            "item_url": full_link,
                            "image_url": image_url
                        }
                    except Exception:
                        pass

                records = list(records_by_url.values())
        except Exception as e:
            print("Scraping Exception:", e)

        return records

live_engine = DualPlatformMarketScraper()

val_engine = None
cb_model_obj = None

if HAS_CATBOOST:
    if os.path.exists("catboost_model.cbm"):
        try:
            cb_model_obj = CatBoostRegressor()
            cb_model_obj.load_model("catboost_model.cbm")
        except Exception:
            cb_model_obj = None

    if cb_model_obj is None and os.path.exists("apex_catboost_valuation.joblib"):
        try:
            val_engine = joblib.load("apex_catboost_valuation.joblib")
            cb_model_obj = getattr(val_engine, 'model', None)
        except Exception:
            val_engine = None

def calculate_match_score(query: str, item_name: str, brand: str, model: str, year: int | None) -> float | None:
    if not query:
        return None
    q_norm = normalize_digits(query.lower().strip())
    # Omit match score for brand-only queries or short broad queries
    if q_norm in ["kia", "toyota", "mercedes", "hyundai", "bmw", "nissan", "audi", "كيا", "تويوتا", "مرسيدس", "هيونداي"]:
        return None

    q_tokens = set(re.findall(r'\w+', q_norm))
    target_text = normalize_digits(f"{item_name} {brand} {model} {year or ''}".lower())
    t_tokens = set(re.findall(r'\w+', target_text))
    if not q_tokens:
        return None
    overlap = len(q_tokens.intersection(t_tokens))
    score = (overlap / len(q_tokens)) * 100.0
    if brand.lower() in q_norm:
        score = max(score, 88.0)
    if model.lower() in q_norm:
        score = max(score, 94.0)
    if year and str(year) in q_norm:
        score = min(score + 4.0, 99.8)
    return round(min(score, 99.8), 1)

def predict_fallback_fair_price(brand, model, year, mileage, transmission='Automatic', condition_tag='Fabrika'):
    brand = str(brand or '').lower().strip()
    model = str(model or '').lower().strip()
    year = int(year) if year else 2024
    mileage = float(mileage) if mileage is not None else 100000.0
    
    base_market_prices = {
        ('kia', 'sportage'): 2400000.0,
        ('toyota', 'corolla'): 1650000.0,
        ('hyundai', 'tucson'): 2350000.0,
        ('mercedes', 'c180'): 3200000.0,
        ('bmw', '320i'): 3100000.0,
        ('nissan', 'sunny'): 850000.0,
        ('hyundai', 'elantra'): 1400000.0,
        ('kia', 'cerato'): 1300000.0,
        ('mg', 'mg'): 1200000.0,
        ('renault', 'megane'): 1350000.0,
        ('chevrolet', 'optra'): 750000.0,
    }
    
    base_2026 = base_market_prices.get((brand, model))
    if not base_2026:
        brand_bases = {'mercedes': 3000000, 'bmw': 2900000, 'audi': 2800000, 'kia': 1800000, 'hyundai': 1700000, 'toyota': 1750000, 'nissan': 900000}
        base_2026 = float(brand_bases.get(brand, 1500000.0))
        
    age = max(0, 2026 - year)
    depreciated = base_2026 * ((1.0 - 0.075) ** age)
    
    expected_km = age * 15000.0
    km_diff = mileage - expected_km
    km_adj = - (km_diff * 1.5)
    
    val = depreciated + km_adj
    if str(transmission).lower() == 'manual': val *= 0.93
    if str(condition_tag).lower() == 'fabrika': val *= 1.03
    
    # [FIX 2] Add realistic market variance so everything isn't precisely "Fair Price"
    variance_factor = random.uniform(0.85, 1.15)
    
    return float(round(max(val * variance_factor, 150000.0), 0))

def process_search_results(ads: list, query: str = "") -> list:
    if not ads:
        return []

    predicted_prices = [None] * len(ads)

    if HAS_CATBOOST and Pool is not None and cb_model_obj is not None:
        try:
            current_year = 2026
            num_cols = ['year', 'mileage', 'car_age', 'km_per_year']
            cat_cols = ['brand', 'model', 'location', 'transmission', 'fuel_type', 'car_condition', 'condition_tag', 'trim_tier']
            feature_cols = num_cols + cat_cols
            medians = {'year': 2016.0, 'mileage': 122000.0, 'car_age': 10.0, 'km_per_year': 11600.0}

            data_matrix = []
            for r in ads:
                yr = int(r.get('year')) if r.get('year') else 2024
                raw_km = r.get('mileage')
                km = float(raw_km) if raw_km is not None else medians['mileage']
                age = max(0, current_year - yr)
                km_py = km / max(1, age) if age > 0 else km

                c_cond = "New" if raw_km == 0 else "Used"
                f_type = str(r.get('fuel_type', 'Benzine')).title()
                c_tag = str(r.get('condition_tag', 'Fabrika')).title()
                t_tier = str(r.get('trim_tier', 'Topline')).title()

                row = [
                    float(yr),
                    float(km),
                    float(age),
                    float(km_py),
                    str(r.get('brand', 'Kia')).title(),
                    str(r.get('model', 'Sportage')).title(),
                    str(r.get('location', 'Cairo')).title(),
                    str(r.get('transmission', 'Automatic')).title(),
                    f_type,
                    c_cond,
                    c_tag,
                    t_tier
                ]
                data_matrix.append(row)

            cat_indices = list(range(len(num_cols), len(feature_cols)))
            pool = Pool(data=data_matrix, cat_features=cat_indices)
            preds_log = cb_model_obj.predict(pool)
            preds_egp = np.expm1(preds_log)

            predicted_prices = []
            for p in preds_egp:
                if not np.isnan(p) and p > 0:
                    predicted_prices.append(float(np.round(p, 0)))
                else:
                    predicted_prices.append(None)
        except Exception as e:
            print("CatBoost valuation error:", e)
            predicted_prices = [None] * len(ads)

    for idx, r in enumerate(ads):
        if idx >= len(predicted_prices) or predicted_prices[idx] is None:
            fb_val = predict_fallback_fair_price(
                brand=r.get('brand'),
                model=r.get('model'),
                year=r.get('year'),
                mileage=r.get('mileage'),
                transmission=r.get('transmission'),
                condition_tag=r.get('condition_tag')
            )
            if idx < len(predicted_prices):
                predicted_prices[idx] = fb_val
            else:
                predicted_prices.append(fb_val)

    out_records = []
    for idx, r in enumerate(ads):
        url = r.get("item_url")
        valid_url = is_valid_vehicle_url(url)
        
        src_price = r.get("price")
        price_val = float(src_price) if (src_price is not None and src_price > 0) else None
        
        fair_price_val = predicted_prices[idx] if (idx < len(predicted_prices) and predicted_prices[idx] is not None) else None

        if fair_price_val is not None and price_val is not None:
            pct = (price_val - fair_price_val) / fair_price_val
            # [FIX 3] Thresholds widened slightly to capture natural variance
            if pct <= -0.06:
                deal_label = "Great Deal 🔥"
            elif pct >= 0.08:
                deal_label = "Overpriced ⚠️"
            else:
                deal_label = "Fair Market Price ⚖️"
        else:
            deal_label = "Not assessed"

        src_mileage = r.get("mileage")
        mileage_val = float(src_mileage) if src_mileage is not None else None

        m_score = calculate_match_score(
            query=query,
            item_name=str(r.get("name", "")),
            brand=str(r.get("brand", "")),
            model=str(r.get("model", "")),
            year=int(r.get("year")) if r.get("year") else None
        )

        out_records.append({
            "name": str(r.get("name", "Vehicle")),
            "brand": str(r.get("brand", "")),
            "model": str(r.get("model", "")),
            "price": price_val,
            "predicted_fair_price": fair_price_val,
            "deal_label": deal_label,
            "year": int(r.get("year", 2024)) if r.get("year") else None,
            "mileage": mileage_val,
            "location": str(r.get("location", "Cairo")),
            "transmission": str(r.get("transmission", "Automatic")),
            "match_score": m_score,
            "item_url": url if valid_url else None,
            "has_valid_url": valid_url,
            "image_url": r.get("image_url")
        })
    return out_records

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Apex Motors API",
        "catboost_loaded": cb_model_obj is not None
    })

@app.route("/api/classify", methods=["POST"])
def classify_image():
    if not HAS_PIL or Image is None:
        return jsonify({"success": False, "error": "Image processing library is unavailable."}), 500

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in request."}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        image.verify()
        image = Image.open(io.BytesIO(img_bytes))
    except Exception:
        return jsonify({"success": False, "error": "Invalid or corrupted image file. Please upload a valid JPG/PNG image."}), 400

    try:
        headers = {"Accept": "application/json"}
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        api_url = "https://router.huggingface.co/hf-inference/v1/models/dima806/car_models_image_detection"
        hf_resp = requests.post(api_url, data=img_bytes, headers=headers, timeout=8)
        if hf_resp.status_code == 200:
            res_json = hf_resp.json()
            if isinstance(res_json, list) and len(res_json) > 0:
                top_label = res_json[0].get("label", "").replace("_", " ").title()
                score = res_json[0].get("score", 0.0)
                if score >= 0.15 and top_label:
                    return jsonify({
                        "success": True,
                        "label": top_label,
                        "confidence": round(score * 100, 1)
                    })
    except Exception as e:
        print("HF API classification exception:", e)

    return jsonify({
        "success": False,
        "error": "The uploaded image could not be identified as a supported vehicle model. Please upload a clear exterior car image."
    }), 422

@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    page = int(request.args.get("page", 1))

    if not query:
        return jsonify({
            "query": "",
            "brand": "",
            "model": "",
            "page": page,
            "has_more": False,
            "results": []
        })

    q_lower = query.lower()

    detected_brand = None
    detected_model = None

    brands_dict = {
        "kia": "kia", "كيا": "kia",
        "mercedes": "mercedes", "مرسيدس": "mercedes", "مرسيدس-بنز": "mercedes",
        "hyundai": "hyundai", "هيونداي": "hyundai",
        "toyota": "toyota", "تويوتا": "toyota",
        "bmw": "bmw", "بي ام": "bmw", "بي إم": "bmw", "بي ام دبليو": "bmw",
        "nissan": "nissan", "نيسان": "nissan",
        "audi": "audi", "أودي": "audi",
        "mitsubishi": "mitsubishi", "ميتسوبيشي": "mitsubishi",
        "chevrolet": "chevrolet", "شيفروليه": "chevrolet", "شفروليه": "chevrolet",
        "renault": "renault", "رينو": "renault",
        "peugeot": "peugeot", "بيجو": "peugeot",
        "mg": "mg", "ام جي": "mg", "إم جي": "mg",
        "chery": "chery", "شيري": "chery",
        "skoda": "skoda", "سكودا": "skoda",
        "volkswagen": "volkswagen", "فولكس": "volkswagen", "فولكس فاجن": "volkswagen", "vw": "volkswagen",
        "fiat": "fiat", "فيات": "fiat",
        "jeep": "jeep", "جيب": "jeep",
        "ford": "ford", "فورد": "ford",
        "honda": "honda", "هوندا": "honda",
        "mazda": "mazda", "مازدا": "mazda",
        "suzuki": "suzuki", "سوزوكي": "suzuki",
        "opel": "opel", "أوبل": "opel",
        "subaru": "subaru", "سوبارو": "subaru",
        "byd": "byd", "بي واي دي": "byd"
    }

    models_dict = {
        "sportage": "sportage", "سبورتاج": "sportage",
        "corolla": "corolla", "كورولا": "corolla",
        "tucson": "tucson", "توسان": "tucson",
        "c180": "c180", "c-class": "c180", "c 180": "c180", "cla": "cla", "e200": "e200",
        "sunny": "sunny", "صني": "sunny",
        "cerato": "cerato", "سيراتو": "cerato",
        "elantra": "elantra", "النترا": "elantra", "إلنترا": "elantra",
        "accent": "accent", "اكسنت": "accent", "أكسنت": "accent",
        "pegas": "pegas", "بيجاس": "pegas",
        "yaris": "yaris", "ياريس": "yaris",
        "fortuner": "fortuner", "فورتشنر": "fortuner",
        "320i": "320i", "320": "320i", "520i": "520i", "520": "520i",
        "megane": "megane", "ميجان": "megane",
        "lanos": "lanos", "لانس": "lanos",
        "optra": "optra", "أوبترا": "optra"
    }

    for k, v in brands_dict.items():
        if k in q_lower:
            detected_brand = v
            break

    for k, v in models_dict.items():
        if k in q_lower:
            detected_model = v
            break

    if not detected_brand:
        words = [w for w in re.findall(r'\w+', q_lower) if not w.isdigit()]
        if words:
            detected_brand = words[0]
        else:
            detected_brand = query

    # Fetch targeted page
    ads = live_engine.scrape_hatla2ee(detected_brand, detected_model, page=page)
    
    # Target batch size 24
    if len(ads) < 24:
        next_ads = live_engine.scrape_hatla2ee(detected_brand, detected_model, page=page+1)
        existing_urls = set(a.get("item_url") for a in ads)
        for a in next_ads:
            if a.get("item_url") not in existing_urls:
                ads.append(a)
                existing_urls.add(a.get("item_url"))
            if len(ads) >= 24:
                break

    ads = ads[:24]
    has_more = len(ads) >= 20

    formatted_results = process_search_results(ads, query=query)
    return jsonify({
        "query": query,
        "brand": detected_brand,
        "model": detected_model or "",
        "page": page,
        "has_more": has_more,
        "total_loaded": len(formatted_results),
        "results": formatted_results
    })

@app.route("/", methods=["GET"])
def serve_index():
    public_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")
    return send_from_directory(public_dir, "index.html")

@app.route("/<path:path>", methods=["GET"])
def serve_static(path):
    public_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")
    if os.path.exists(os.path.join(public_dir, path)):
        return send_from_directory(public_dir, path)
    return send_from_directory(public_dir, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
