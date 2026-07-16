"""
Quick integration test for LM Studio + Gemma with the scratchpad inference engine.
Run after starting LM Studio with a Gemma model loaded.

Usage:
    $env:SCRATCHPAD_LLM_BACKEND = "lmstudio"
    $env:SCRATCHPAD_MODEL_NAME = "gemma-3-4b-it"  # or whatever your model is named
    python test_lmstudio_integration.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Verify env
backend = os.getenv("SCRATCHPAD_LLM_BACKEND", "not set")
model = os.getenv("SCRATCHPAD_MODEL_NAME", "not set")
print(f"[CONFIG] backend={backend}, model={model}")

# Test 1: basic connectivity to LM Studio server
print("\n=== Test 1: Server connectivity ===")
try:
    import httpx
    resp = httpx.get("http://localhost:1234/v1/models", timeout=10)
    resp.raise_for_status()
    models = resp.json()
    available = [m["id"] for m in models.get("data", [])]
    print(f"LM Studio models: {available}")
    if not available:
        print("[WARN] No models loaded in LM Studio. Load Gemma first, then restart the server.")
except Exception as e:
    print(f"[FAIL] Cannot reach LM Studio server: {e}")
    print("Make sure LM Studio is running and the Local Server is active.")
    sys.exit(1)

# Test 2: raw chat completion (no structured output)
print("\n=== Test 2: Raw chat completion ===")
try:
    resp = httpx.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, LM Studio!' and nothing else."}
            ],
            "temperature": 0.1,
            "max_tokens": 50
        },
        timeout=60
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    print(f"[PASS] Raw chat response: {content[:100]}")
except Exception as e:
    print(f"[FAIL] Raw chat failed: {e}")
    sys.exit(1)

# Test 3: structured output via response_format
print("\n=== Test 3: Structured output (response_format) ===")
try:
    resp = httpx.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You extract structured facts from text."},
                {"role": "user", "content": "From this text: 'The auth service calls the database on port 5432.' "
                 "Extract a JSON object with fields: source_entity, relationship, target_entity."}
            ],
            "temperature": 0.1,
            "max_tokens": 200,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "FactExtraction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "source_entity": {"type": "string"},
                            "relationship": {"type": "string"},
                            "target_entity": {"type": "string"}
                        },
                        "required": ["source_entity", "relationship", "target_entity"]
                    }
                }
            }
        },
        timeout=120
    )
    resp.raise_for_status()
    raw = resp.json()
    content = raw["choices"][0]["message"]["content"]
    print(f"[PASS] Structured response: {content[:200]}")
    import json
    parsed = json.loads(content)
    print(f"       Parsed fields: source={parsed.get('source_entity')}, "
          f"rel={parsed.get('relationship')}, target={parsed.get('target_entity')}")
except Exception as e:
    print(f"[FAIL] Structured output failed: {e}")
    sys.exit(1)

# Test 4: full inference engine via UniversalInferenceEngine
print("\n=== Test 4: UniversalInferenceEngine with LM Studio backend ===")
try:
    # Use "lmstudio" backend and the actual model we discovered
    actual_model = available[0]  # "google/gemma-4-e4b"
    os.environ["SCRATCHPAD_LLM_BACKEND"] = "lmstudio"
    os.environ["SCRATCHPAD_MODEL_NAME"] = actual_model

    # Import and instantiate AFTER env vars are set (env read at __init__)
    from inference import UniversalInferenceEngine
    from schema import GraphTriplet

    engine = UniversalInferenceEngine()
    print(f"[CONFIG] Engine initialized: backend={engine.backend_type}, model={engine.model_name}")

    result = engine.generate_structured(
        prompt="From this text: 'The auth service calls the database on port 5432.' "
               "Extract exactly one fact as a JSON object. The citation_quote should be the exact phrase "
               "from the text that supports this fact.",
        system_prompt="You are a precise factual extraction system. Output ONLY valid JSON.",
        response_schema=GraphTriplet
    )
    print(f"[PASS] Engine returned: {result}")
    print(f"       source_entity={result.source_entity}, relationship={result.relationship}, "
          f"target_entity={result.target_entity}, citation={result.citation_quote}")
except Exception as e:
    print(f"[FAIL] Inference engine failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== All tests passed! ===")
