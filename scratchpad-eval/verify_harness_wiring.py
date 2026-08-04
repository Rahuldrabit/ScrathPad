"""
Verify the harness's WIRING against the real ScratchpadPoweredLLM class -
construction, record_observation -> extract -> commit -> view, all through
the actual repo code. The only thing mocked is the LLM call itself (which
needs a live local backend I don't have in this sandbox); everything
around it - session init, the verification gate, canonicalization, the
bounded view compiler, the citation gate - runs for real.
"""
import sys, os
sys.path.insert(0, "src")

import database
database.DB_PATH = "/tmp/_harness_wiring_test.db"
if os.path.exists(database.DB_PATH):
    os.remove(database.DB_PATH)
database.initialize_database()

import inference
from schema import GraphTriplet
from scratchpad_agent import ScratchpadPoweredLLM, FactExtraction

# Mock only the LLM-facing extraction call. It returns one real, correctly
# cited fact and demonstrates the citation gate still runs for real: the
# raw_chunk passed to record_observation must actually contain this exact
# quote or commit_page_data_to_sqlite will reject it, same as any real call.
SAMPLE_CHUNK = "auth-service connection pool has not released a connection in the last 6 minutes."

def mock_generate_structured(self, prompt, system_prompt, response_schema):
    if response_schema is FactExtraction:
        return FactExtraction(
            triplets=[GraphTriplet(
                direction_check="[AUTH_SERVICE] -> [connects_to] -> [CONNECTION_POOL]",
                source_entity="AUTH_SERVICE",
                source_type="SERVICE",
                relationship="connects_to",
                target_entity="CONNECTION_POOL",
                target_type="SERVICE",
                citation_quote="auth-service connection pool has not released a connection",
            )],
            resolved_variables=[],
            new_missing_variables=["GUARD_ROTATION_SCHEDULE"],
        )
    raise AssertionError(f"Unexpected schema requested: {response_schema}")

inference.UniversalInferenceEngine.generate_structured = mock_generate_structured

# --- Construct (auto_plan=False so no LLM call happens at init) ---
agent = ScratchpadPoweredLLM(
    goal="Investigate auth-service connection pool exhaustion",
    context_size=4000,
    session_id="wiring_test",
    backend="lmstudio",
    auto_plan=False,
)
print("[OK] ScratchpadPoweredLLM constructed without a live LLM backend")

# --- record_observation: extraction (mocked) -> citation gate (real) -> commit (real) ---
committed = agent.record_observation(SAMPLE_CHUNK, source="incident_log")
print(f"[OK] record_observation committed {committed} triplet(s)")
assert committed == 1, f"expected 1 committed triplet, got {committed}"

# --- view compilation (real, token-bounded) ---
view = agent.get_scratchpad_view()
print("\n--- compiled scratchpad view ---")
print(view)
assert "AUTH_SERVICE" in view, "committed fact should appear in the view"
assert "GUARD_ROTATION_SCHEDULE" in view, "unresolved variable should appear in the view"
print("\n[OK] committed fact and unresolved variable both present in the compiled view")

os.remove(database.DB_PATH)
print("\nHARNESS WIRING VERIFIED — the only missing piece for a real run is a live LLM endpoint.")
