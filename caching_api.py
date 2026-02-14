import hashlib
import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from collections import OrderedDict
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# CONFIGURATION
# =============================

MAX_CACHE_SIZE = 1500
TTL_HOURS = 24
SIMILARITY_THRESHOLD = 0.95
MODEL_COST_PER_MILLION = 0.40
AVG_TOKENS = 300

# =============================
# DATA STRUCTURES
# =============================

cache = OrderedDict()
embeddings_store = {}
analytics = {
    "totalRequests": 0,
    "cacheHits": 0,
    "cacheMisses": 0
}

# =============================
# REQUEST MODEL
# =============================

class QueryRequest(BaseModel):
    query: str
    application: str

# =============================
# UTILITIES
# =============================

def md5_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def fake_llm_call(query):
    time.sleep(0.8)  # simulate API latency
    return f"Moderation result for: {query}"

def generate_embedding(text):
    np.random.seed(abs(hash(text)) % (10**8))
    return np.random.rand(384)

def is_expired(entry_time):
    return datetime.utcnow() > entry_time + timedelta(hours=TTL_HOURS)

def evict_if_needed():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)

# =============================
# MAIN ENDPOINT
# =============================

@app.post("/")
def handle_query(req: QueryRequest):

    start_time = time.time()
    analytics["totalRequests"] += 1

    query = req.query
    key = md5_hash(query)

    # 1️⃣ Exact Match Check
    if key in cache:
        entry = cache[key]
        if not is_expired(entry["timestamp"]):
            analytics["cacheHits"] += 1
            cache.move_to_end(key)
            latency = int((time.time() - start_time) * 1000)
            return {
                "answer": entry["response"],
                "cached": True,
                "latency": latency,
                "cacheKey": key
            }
        else:
            del cache[key]

    # 2️⃣ Semantic Check
    new_embedding = generate_embedding(query)

    for stored_key, entry in cache.items():
        if is_expired(entry["timestamp"]):
            continue
        similarity = cosine_similarity(
            [new_embedding],
            [entry["embedding"]]
        )[0][0]
        if similarity > SIMILARITY_THRESHOLD:
            analytics["cacheHits"] += 1
            cache.move_to_end(stored_key)
            latency = int((time.time() - start_time) * 1000)
            return {
                "answer": entry["response"],
                "cached": True,
                "latency": latency,
                "cacheKey": stored_key
            }

    # 3️⃣ Cache Miss → Call LLM
    analytics["cacheMisses"] += 1
    response = fake_llm_call(query)

    cache[key] = {
        "response": response,
        "embedding": new_embedding,
        "timestamp": datetime.utcnow()
    }

    cache.move_to_end(key)
    evict_if_needed()

    latency = int((time.time() - start_time) * 1000)

    return {
        "answer": response,
        "cached": False,
        "latency": latency,
        "cacheKey": key
    }

# =============================
# ANALYTICS ENDPOINT
# =============================

@app.get("/analytics")
def get_analytics():

    hits = analytics["cacheHits"]
    misses = analytics["cacheMisses"]
    total = analytics["totalRequests"]

    hit_rate = hits / total if total > 0 else 0
    savings_tokens = hits * AVG_TOKENS
    cost_savings = (savings_tokens / 1_000_000) * MODEL_COST_PER_MILLION
    baseline_cost = (total * AVG_TOKENS / 1_000_000) * MODEL_COST_PER_MILLION
    savings_percent = (cost_savings / baseline_cost) * 100 if baseline_cost else 0

    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": total,
        "cacheHits": hits,
        "cacheMisses": misses,
        "cacheSize": len(cache),
        "costSavings": round(cost_savings, 2),
        "savingsPercent": round(savings_percent, 2),
        "strategies": [
            "exact match",
            "semantic similarity",
            "LRU eviction",
            "TTL expiration"
        ]
    }

