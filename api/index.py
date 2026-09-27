import os
import re
import io
import sys
import json
import time
import unicodedata
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

DETAIL_URL_PATTERN = re.compile(r'/(?:car|new-car)/[^\?#]*?\d{5,}$', re.IGNORECASE)
ARABIC_TO_ENGLISH_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


# ==============================================================================
# ARABIC NLP LAYER (restored from notebook)
# ==============================================================================
# These synonym maps and helpers bring back the dialect-aware understanding
# that was present in the notebook (budget extraction, location relaxation,
# condition/trim synonyms) but was missing from the earlier production code.

BODY_SYNONYMS = {"suv": "SUV", "دفع رباعي": "SUV", "جيب": "SUV", "sedan": "Sedan",
                 "سيدان": "Sedan", "hatchback": "Hatchback", "هاتشباك": "Hatchback"}

TRANS_SYNONYMS = {"automatic": "Automatic", "اوتوماتيك": "Automatic", "أوتوماتيك": "Automatic",
                  "auto": "Automatic", "manual": "Manual", "مانيوال": "Manual", "يدوي": "Manual"}

FUEL_SYNONYMS = {"بنزين": "Benzine", "gas": "Gas", "غاز": "Gas",
                 "كهرباء": "Electric", "electric": "Electric", "هايبرد": "Hybrid", "hybrid": "Hybrid"}

PAINT_SYNONYMS = {"فابريكا": "Fabrika", "فابريك": "Fabrika", "فبريكة": "Fabrika", "فبريك": "Fabrika",
                  "وكالة": "Fabrika", "بدون جرام": "Fabrika", "زيرو": "Fabrika",
                  "رش حزام": "Belt_Repaint", "راشة حزام": "Belt_Repaint", "حزام نظافة": "Belt_Repaint",
                  "حادث": "Accident_Repaired", "دواخل": "Accident_Repaired", "مرمات": "Accident_Repaired"}

TRIM_SYNONYMS = {"اعلى فئة": "Topline", "أعلى فئة": "Topline", "topline": "Topline",
                 "top line": "Topline", "بانوراما": "Topline", "كاملة": "Topline",
                 "هاي لاين": "Highline", "highline": "Highline", "high line": "Highline",
                 "بيز لاين": "Baseline", "baseline": "Baseline", "فاضية": "Baseline", "عادي": "Baseline"}

CITY_MAP = {
    "القاهرة": "Cairo", "cairo": "Cairo",
    "التجمع": "Tagamo3", "tagamo3": "Tagamo3", "القاهرة الجديدة": "New Cairo",
    "اسكندرية": "Alexandria", "الإسكندرية": "Alexandria", "alexandria": "Alexandria",
    "الجيزة": "Giza", "giza": "Giza",
    "زايد": "Sheikh Zayed", "الشيخ زايد": "Sheikh Zayed", "sheikh zayed": "Sheikh Zayed",
    "اكتوبر": "6th of October", "٦ اكتوبر": "6th of October",
    "مدينة نصر": "Nasr City", "مصر الجديدة": "Heliopolis", "المعادي": "Maadi",
    "مهندسين": "Mohandessin", "المهندسين": "Mohandessin",
    "الشروق": "Al Shorouk", "شروق": "Al Shorouk"
}

ARABIC_BRAND_MAP = {
    "كيا": "kia", "مرسيدس": "mercedes", "هيونداي": "hyundai", "تويوتا": "toyota",
    "بي ام دبليو": "bmw", "بي ام": "bmw", "نيسان": "nissan", "أودي": "audi",
    "ميتسوبيشي": "mitsubishi", "شيفروليه": "chevrolet", "شفروليه": "chevrolet",
    "رينو": "renault", "بيجو": "peugeot", "ام جي": "mg", "إم جي": "mg",
    "شيري": "chery", "سكودا": "skoda", "فولكس فاجن": "volkswagen", "فولكس": "volkswagen",
    "فيات": "fiat", "جيب": "jeep", "فورد": "ford", "هوندا": "honda",
    "مازدا": "mazda", "سوزوكي": "suzuki", "أوبل": "opel", "اوبل": "opel",
    "سوبارو": "subaru", "بي واي دي": "byd", "بورش": "porsche", "بورشه": "porsche"
}

ARABIC_MODEL_MAP = {
    "سبورتاج": "sportage", "كورولا": "corolla", "توسان": "tucson",
    "سي 180": "c180", "سي180": "c180", "سي ال ايه": "cla",
    "صني": "sunny", "سيراتو": "cerato", "النترا": "elantra", "إلنترا": "elantra",
    "اكسنت": "accent", "أكسنت": "accent", "بيجاس": "pegas", "ياريس": "yaris",
    "فورتشنر": "fortuner", "ميجان": "megane", "لانس": "lanos", "أوبترا": "optra",
    "بي ار زد": "brz", "911": "911"
}


def normalize_arabic(text: str) -> str:
    """Normalizes Arabic dialect variants and converts spoken-number shorthand,
    exactly like the notebook's normalize_arabic() so the same phrasing
    ('800 الف', 'ونص', etc.) is understood in production."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text.lower().strip())
    t = t.translate(ARABIC_TO_ENGLISH_DIGITS)
    t = re.sub(r"[إأآا]", "ا", t)
    t = re.sub(r"ة\b", "ه", t)
    t = re.sub(r"ى\b", "ي", t)
    t = t.replace("ونص", ".5").replace("وربع", ".25").replace("وتلت", ".33")
    t = t.replace("باكو", " الف").replace("ارنب", " مليون").replace("أرنب", " مليون")
    return t


def extract_budget(text: str):
    """Restores the notebook's budget parser: handles ranges ('من X ل Y'),
    upper bounds ('تحت X'), and single point budgets ('بـ X')."""
    def scale(val, unit):
        if not unit:
            return val if val >= 10000 else val * 1_000_000
        u = unit.lower()
        if u in ("m", "مليون"):
            return val * 1_000_000
        if u in ("k", "الف", "ألف"):
            return val * 1_000
        return val

    m = re.search(
        r'(?:من\s*)?(\d+(?:\.\d+)?)\s*(m|مليون|k|الف)?\s*(?:-|to|حتى|لحد|الى|لـ)\s*'
        r'(\d+(?:\.\d+)?)\s*(m|مليون|k|الف)?', text)
    if m:
        u_fin = m.group(4) or m.group(2)
        lo = scale(float(m.group(1)), u_fin)
        hi = scale(float(m.group(3)), u_fin)
        return min(lo, hi), max(lo, hi)

    m = re.search(r'(?:تحت|اقل من|حتى|في حدود|سقف)\s*(\d+(?:\.\d+)?)\s*(m|مليون|k|الف)?', text)
    if m:
        return None, scale(float(m.group(1)), m.group(2))

    m = re.search(r'(?:بـ|ب|معايا)\s*(\d+(?:\.\d+)?)\s*(m|مليون|k|الف)', text)
    if m:
        v = scale(float(m.group(1)), m.group(2))
        return v * 0.85, v * 1.15

    return None, None


def parse_query_filters(query: str):
    """Full dialect-aware parse of a free-text query: brand, model, location,
    budget range, condition tag, trim tier, transmission and fuel type.
    This is the piece that made the notebook 'smart' and was entirely
    missing from the earlier production /api/search."""
    norm_q = normalize_arabic(query)
    filters = {
        "brand": None, "model": None, "location": None,
        "min_price": None, "max_price": None,
        "condition_tag": None, "trim_tier": None,
        "transmission": None, "fuel_type": None, "body_type": None,
    }

    p_min, p_max = extract_budget(norm_q)
    filters["min_price"] = p_min
    filters["max_price"] = p_max

    for col, syns in [("body_type", BODY_SYNONYMS), ("transmission", TRANS_SYNONYMS),
                      ("fuel_type", FUEL_SYNONYMS), ("condition_tag", PAINT_SYNONYMS),
                      ("trim_tier", TRIM_SYNONYMS)]:
        for k, val in syns.items():
            if k in norm_q:
                filters[col] = val
                break

    for ar, en in sorted(ARABIC_BRAND_MAP.items(), key=lambda x: -len(x[0])):
        if ar in norm_q:
            filters["brand"] = en
            break
    if not filters["brand"]:
        for en in set(ARABIC_BRAND_MAP.values()):
            if en in norm_q:
                filters["brand"] = en
                break

    for ar, en in sorted(ARABIC_MODEL_MAP.items(), key=lambda x: -len(x[0])):
        if ar in norm_q:
            filters["model"] = en
            break
    if not filters["model"]:
        for en in set(ARABIC_MODEL_MAP.values()):
            if en in norm_q:
                filters["model"] = en
                break

    for ar, en in sorted(CITY_MAP.items(), key=lambda x: -len(x[0])):
        if ar.lower() in norm_q:
            filters["location"] = en
            break

    return filters


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


# ==============================================================================
# SCRAPER: Hatla2ee + Dubizzle/OLX (dual platform restored from notebook)
# ==============================================================================

class DualPlatformMarketScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        self.base_url = "https://eg.hatla2ee.com"
        self.dubizzle_url = "https://www.dubizzle.com.eg"

    # ---- Hatla2ee ----
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

                cards = soup.find_all(lambda tag: tag.name in ['div', 'article', 'section']
                                       and tag.get('class') and 'bg-card' in tag.get('class'))
                if not cards:
                    cards = soup.find_all(lambda tag: tag.name in ['div', 'article', 'section']
                                           and tag.get('class')
                                           and any('unit' in c.lower() or 'card' in c.lower() for c in tag.get('class')))

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

                        condition_tag = "Fabrika" if "فابريكا" in card_text_space else "Normal"
                        trim_tier = "Topline" if any(k in card_text_space for k in ["اعلى فئة", "بانوراما", "توب لاين"]) else "Standard"

                        tokens = [t.strip() for t in card_text_bar.split('|') if t.strip()]
                        location = "Cairo"
                        known_locs = ["القاهرة", "الجيزة", "الإسكندرية", "التجمع", "المهندسين", "دمياط",
                                      "منوفية", "الشرقية", "الدقهلية", "الغربية", "أسيوط", "سوهاج",
                                      "المنيا", "بني سويف", "الفيوم", "إسماعيلية", "السويس", "بورسعيد"]
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
                            "trim_tier": trim_tier,
                            "source": "Hatla2ee",
                            "item_url": full_link,
                            "image_url": image_url
                        }
                    except Exception:
                        pass

                records = list(records_by_url.values())
        except Exception as e:
            print("Hatla2ee scraping exception:", e)

        return records

    # ---- Dubizzle / OLX (restored from notebook) ----
    def scrape_dubizzle_olx(self, brand: str, model: str = None) -> list:
        records = []
        if not brand:
            return []

        target_b = brand.lower().strip()
        clean_m = re.sub(r"\b(class|series|sedan|suv|coupe)\b", "", model or "", flags=re.IGNORECASE).strip()
        query_str = f"{brand} {clean_m}".strip() if clean_m else brand
        search_url = f"{self.dubizzle_url}/vehicles/cars-for-sale/?q={requests.utils.quote(query_str)}"

        try:
            resp = self.session.get(search_url, headers=self.headers, timeout=12)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                next_data = soup.select_one("script#__NEXT_DATA__")
                if next_data and next_data.string:
                    try:
                        payload = json.loads(next_data.string)
                        raw_ads = (payload.get("props", {})
                                          .get("pageProps", {})
                                          .get("initialState", {})
                                          .get("feed", {})
                                          .get("data", []))
                        for item in raw_ads:
                            title = (item.get("title") or item.get("name") or "").strip()
                            t_lower = title.lower()
                            if target_b not in t_lower:
                                continue

                            price_val = item.get("price", {}).get("value")
                            if not price_val:
                                continue

                            loc_name = item.get("location", {}).get("name", "Cairo")
                            ad_id = item.get("id", "")
                            ad_url = f"{self.dubizzle_url}/ad/{ad_id}" if ad_id else None
                            y_match = re.search(r"\b(19\d{2}|20\d{2})\b", title)
                            year = int(y_match.group(1)) if y_match else 2024

                            image_url = None
                            imgs = item.get("images") or item.get("photos")
                            if isinstance(imgs, list) and imgs:
                                first = imgs[0]
                                image_url = first.get("url") if isinstance(first, dict) else first

                            records.append({
                                "name": title,
                                "brand": brand.title(),
                                "model": model.title() if model else "Model",
                                "price": float(price_val),
                                "year": year,
                                "mileage": None,
                                "location": loc_name,
                                "transmission": "Automatic",
                                "fuel_type": "Benzine",
                                "car_condition": "Used",
                                "condition_tag": "Fabrika" if any(k in title for k in ["فابريك", "فبريك", "وكالة", "زيرو"]) else "Normal",
                                "trim_tier": "Topline" if any(k in title for k in ["اعلى فئة", "توب لاين", "topline", "بانوراما"]) else "Standard",
                                "source": "OLX / Dubizzle",
                                "item_url": ad_url,
                                "image_url": image_url
                            })
                    except Exception as e:
                        print("Dubizzle JSON parse exception:", e)
        except Exception as e:
            print("Dubizzle scraping exception:", e)

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
    if q_norm in ["kia", "toyota", "mercedes", "hyundai", "bmw", "nissan", "audi",
                  "كيا", "تويوتا", "مرسيدس", "هيونداي"]:
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
        brand_bases = {'mercedes': 3000000, 'bmw': 2900000, 'audi': 2800000, 'kia': 1800000,
                       'hyundai': 1700000, 'toyota': 1750000, 'nissan': 900000}
        base_2026 = float(brand_bases.get(brand, 1500000.0))

    age = max(0, 2026 - year)
    depreciated = base_2026 * ((1.0 - 0.075) ** age)

    expected_km = age * 15000.0
    km_diff = mileage - expected_km
    km_adj = -(km_diff * 1.5)

    val = depreciated + km_adj
    if str(transmission).lower() == 'manual':
        val *= 0.93
    if str(condition_tag).lower() == 'fabrika':
        val *= 1.03
    return float(round(max(val, 150000.0), 0))


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
                    float(yr), float(km), float(age), float(km_py),
                    str(r.get('brand', 'Kia')).title(),
                    str(r.get('model', 'Sportage')).title(),
                    str(r.get('location', 'Cairo')).title(),
                    str(r.get('transmission', 'Automatic')).title(),
                    f_type, c_cond, c_tag, t_tier
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
                brand=r.get('brand'), model=r.get('model'), year=r.get('year'),
                mileage=r.get('mileage'), transmission=r.get('transmission'),
                condition_tag=r.get('condition_tag')
            )
            if idx < len(predicted_prices):
                predicted_prices[idx] = fb_val
            else:
                predicted_prices.append(fb_val)

    out_records = []
    for idx, r in enumerate(ads):
        url = r.get("item_url")
        valid_url = is_valid_vehicle_url(url) if url else False

        src_price = r.get("price")
        price_val = float(src_price) if (src_price is not None and src_price > 0) else None

        fair_price_val = predicted_prices[idx] if (idx < len(predicted_prices) and predicted_prices[idx] is not None) else None

        if fair_price_val is not None and price_val is not None:
            pct = (price_val - fair_price_val) / fair_price_val
            if pct <= -0.05:
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
            query=query, item_name=str(r.get("name", "")),
            brand=str(r.get("brand", "")), model=str(r.get("model", "")),
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
            "condition_tag": str(r.get("condition_tag", "Normal")),
            "trim_tier": str(r.get("trim_tier", "Standard")),
            "source": str(r.get("source", "Hatla2ee")),
            "match_score": m_score,
            "item_url": url if valid_url else url,  # OLX links pass through even without the Hatla2ee-only regex
            "has_valid_url": valid_url or (url is not None and "dubizzle" in url),
            "image_url": r.get("image_url")
        })
    return out_records


def apply_filters_and_relaxation(records: list, filters: dict):
    """Restores the notebook's budget alignment + location relaxation logic:
    filter by location/budget when possible, but if nothing matches, relax
    the constraint and tell the user why (instead of silently returning
    nothing, which is what plain filtering would do)."""
    notes = []
    working = records

    if filters.get("location"):
        loc = filters["location"].lower()
        loc_matches = [r for r in working if loc in str(r.get("location", "")).lower()]
        if loc_matches:
            working = loc_matches
        else:
            car_tag = f"{filters.get('brand', '').title()} {filters.get('model', '').title()}".strip()
            notes.append(f"لم تتوفر سيارات مطابقة في ({filters['location']})، تم توسيع النطاق لأقرب سيارات {car_tag}.")

    if filters.get("min_price") or filters.get("max_price"):
        lo = filters.get("min_price") or 0
        hi = filters.get("max_price") or float("inf")
        budget_matches = [r for r in working if r.get("price") and lo <= r["price"] <= hi]
        if budget_matches:
            working = budget_matches
        else:
            target = filters.get("max_price") or filters.get("min_price")
            working = sorted(working, key=lambda r: abs((r.get("price") or target) - target))
            notes.append(f"الميزانية المطلوبة ({target:,.0f} EGP) لا تتطابق تمامًا مع أسعار السوق المتاحة، "
                         f"تم ترتيب النتائج من الأقرب لميزانيتك.")

    if filters.get("condition_tag"):
        tag_matches = [r for r in working if r.get("condition_tag") == filters["condition_tag"]]
        if tag_matches:
            working = tag_matches

    if filters.get("transmission"):
        trans_matches = [r for r in working if r.get("transmission") == filters["transmission"]]
        if trans_matches:
            working = trans_matches

    return working, notes


# ==============================================================================
# VISION: HuggingFace API with top-5 disambiguation + label cleaning
# (restored from the notebook's classify_car_image / predict_top5_cars_from_image)
# ==============================================================================

DISAMBIGUATION_GROUPS = [
    {"subaru", "brz", "toyota", "gr86", "gt86", "86"},
]

NOISE_SUFFIX_PATTERN = re.compile(r"\b(class|series|sedan|suv|coupe)\b", re.IGNORECASE)


def clean_vision_label(raw_label: str) -> tuple:
    """Splits a raw HF label like 'Mercedes Benz Cla' into (brand, model) and
    strips generic noise words, mirroring the notebook's classify_car_image()."""
    raw = raw_label.replace("_", " ").strip()
    label_lower = raw.lower()

    multiword_brands = ["alfa romeo", "land rover", "aston martin", "mercedes benz", "rolls royce"]
    brand_rename = {"mercedes benz": "mercedes", "vw": "volkswagen", "chevy": "chevrolet"}

    detected_brand = None
    detected_model = ""
    for mb in multiword_brands:
        if label_lower.startswith(mb):
            detected_brand = brand_rename.get(mb, mb)
            detected_model = raw[len(mb):].strip()
            break

    if not detected_brand:
        tokens = raw.split()
        if tokens:
            first_token = tokens[0].lower()
            detected_brand = brand_rename.get(first_token, first_token)
            detected_model = " ".join(tokens[1:]) if len(tokens) > 1 else ""

    clean_model = NOISE_SUFFIX_PATTERN.sub("", detected_model).strip()
    return (detected_brand or "").title(), (clean_model or "").title()


def classify_car_image_via_api(img_bytes: bytes, max_retries: int = 2) -> dict:
    """Calls the HuggingFace Inference API, retries on transient failures,
    and applies the same top-5 disambiguation + suffix-cleaning the notebook
    used to do locally with the transformers model."""
    headers = {"Accept": "application/json"}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    api_url = f"https://router.huggingface.co/hf-inference/v1/models/{VISION_MODEL_NAME}"

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(api_url, data=img_bytes, headers=headers, timeout=15)
            if resp.status_code == 200:
                candidates = resp.json()
                if isinstance(candidates, list) and candidates:
                    labels_combined = " ".join(c.get("label", "").lower() for c in candidates)

                    top = candidates[0]
                    selected_label = top.get("label", "")
                    selected_score = top.get("score", 0.0)

                    for group in DISAMBIGUATION_GROUPS:
                        if any(k in labels_combined for k in group):
                            for c in candidates:
                                if any(k in c.get("label", "").lower() for k in group):
                                    selected_label = c.get("label", "")
                                    selected_score = c.get("score", 0.0)
                                    break
                            break

                    brand, model = clean_vision_label(selected_label)
                    if selected_score >= 0.15 and brand:
                        return {
                            "success": True,
                            "brand": brand,
                            "model": model,
                            "label": f"{brand} {model}".strip(),
                            "confidence": round(selected_score * 100, 1),
                            "candidates": [
                                {"label": c.get("label"), "confidence": round(c.get("score", 0) * 100, 1)}
                                for c in candidates[:5]
                            ]
                        }
                    last_error = "low_confidence"
                else:
                    last_error = "empty_response"
            elif resp.status_code == 503:
                # Model is loading on HF's side; wait briefly and retry.
                last_error = "model_loading"
                time.sleep(2)
                continue
            else:
                last_error = f"http_{resp.status_code}"
        except requests.exceptions.Timeout:
            last_error = "timeout"
        except Exception as e:
            last_error = str(e)

        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))

    return {"success": False, "error": last_error or "unknown_error"}


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Apex Motors API",
        "catboost_loaded": cb_model_obj is not None,
        "vision_backend": "huggingface_api",
        "sources": ["Hatla2ee", "OLX/Dubizzle"]
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
    except Exception:
        return jsonify({"success": False, "error": "Invalid or corrupted image file. Please upload a valid JPG/PNG image."}), 400

    result = classify_car_image_via_api(img_bytes)
    if result.get("success"):
        return jsonify(result)

    return jsonify({
        "success": False,
        "error": "The uploaded image could not be identified as a supported vehicle model. "
                 "Please upload a clear exterior car image.",
        "detail": result.get("error")
    }), 422


@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    page = int(request.args.get("page", 1))

    if not query:
        return jsonify({
            "query": "", "brand": "", "model": "",
            "page": page, "has_more": False, "results": [], "notices": []
        })

    filters = parse_query_filters(query)
    detected_brand = filters["brand"]
    detected_model = filters["model"]

    if not detected_brand:
        words = [w for w in re.findall(r'\w+', normalize_arabic(query)) if not w.isdigit()]
        detected_brand = words[0] if words else query

    # Dual-platform scraping, restored from the notebook.
    hatla_ads = live_engine.scrape_hatla2ee(detected_brand, detected_model, page=page)
    if len(hatla_ads) < 20:
        next_ads = live_engine.scrape_hatla2ee(detected_brand, detected_model, page=page + 1)
        existing_urls = set(a.get("item_url") for a in hatla_ads)
        for a in next_ads:
            if a.get("item_url") not in existing_urls:
                hatla_ads.append(a)
                existing_urls.add(a.get("item_url"))

    olx_ads = []
    try:
        olx_ads = live_engine.scrape_dubizzle_olx(detected_brand, detected_model)
    except Exception as e:
        print("OLX fetch skipped:", e)

    ads = hatla_ads + olx_ads
    ads = ads[:30]
    has_more = len(hatla_ads) >= 20

    formatted_results = process_search_results(ads, query=query)
    filtered_results, relaxation_notes = apply_filters_and_relaxation(formatted_results, filters)

    return jsonify({
        "query": query,
        "brand": detected_brand,
        "model": detected_model or "",
        "filters": {k: v for k, v in filters.items() if v is not None},
        "page": page,
        "has_more": has_more,
        "total_loaded": len(filtered_results),
        "notices": relaxation_notes,
        "results": filtered_results
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
