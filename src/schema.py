"""
Pydantic schemas for the scratchpad middleware.

Layer 1 (closed relationship vocabulary) and in-schema CoT (direction_check)
are enforced here at the schema level so they apply across all four
inference backends, not just at the engine-level filter.
"""
import re
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────
# Closed type vocabulary — model picks from this literal, never freeform
# ─────────────────────────────────────────────────────────────────────────
EntityType = Literal[
    "SERVICE",      # microservice, daemon, server process
    "FILE",         # source file path
    "TABLE",        # DB table / collection
    "CONFIG_KEY",   # config key, env var name, secret name
    "FUNCTION",     # named function in code
    "QUEUE",        # message queue, stream, topic
    "PROTOCOL",     # network protocol identifier
    "ENV_VAR",      # environment variable
    "CACHE",        # cache layer (Redis, Memcached, in-memory)
    "DATABASE",     # database engine / cluster
]


# ─────────────────────────────────────────────────────────────────────────
# Per-EntityType regex allowlist
#
# The EntityType literal only checks that source_type/target_type is one of
# the allowed strings. It does NOT check that the entity string itself
# looks like its declared type. A source_type="SERVICE" with
# source_entity="db.query('SELECT ...')" is clearly wrong but previously
# passed the schema. These patterns reject that class of mistake.
#
# Patterns are deliberately permissive — they exist to catch obviously
# malformed entities (embedded code, SQL, raw prose), not to enforce a
# strict naming convention. The shapes they expect:
#   SERVICE/TABLE/CACHE/DATABASE/ENV_VAR  → UPPER_SNAKE_CASE identifier
#   FILE                                   → a path with an extension
#   FUNCTION                               → lowerCamelCase / snake_case
#   CONFIG_KEY                             → dotted or UPPER_SNAKE key
#   QUEUE                                  → topic/stream name
#   PROTOCOL                               → a known protocol literal
# ─────────────────────────────────────────────────────────────────────────
ENTITY_PATTERNS: Dict[str, "re.Pattern"] = {
    "SERVICE":    re.compile(r"^[A-Z][A-Z0-9_]{2,40}$"),
    "FILE":       re.compile(r"^[\w./\\-]+\.\w{1,8}$"),
    "TABLE":      re.compile(r"^[A-Z][A-Z0-9_]{2,40}$"),
    "CONFIG_KEY": re.compile(r"^[A-Z][A-Z0-9_.]{2,80}$"),
    "FUNCTION":   re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{1,60}$"),
    "QUEUE":      re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{1,60}$"),
    "PROTOCOL":   re.compile(r"^(https?|grpc|amqp|tcp|udp|ws|wss|mssql|postgres(?:ql)?|mysql|redis)$"),
    "ENV_VAR":    re.compile(r"^[A-Z][A-Z0-9_]{2,60}$"),
    "CACHE":      re.compile(r"^[A-Z][A-Z0-9_]{2,40}$"),
    "DATABASE":   re.compile(r"^[A-Z][A-Z0-9_]{2,40}$"),
}


def validate_entity_shape(entity: str, entity_type: str) -> bool:
    """
    Returns True if `entity` matches the expected shape for `entity_type`.
    Unknown types (not in ENTITY_PATTERNS) are accepted — this layer only
    rejects obvious mismatches, it does not gate on novel type names.
    """
    pattern = ENTITY_PATTERNS.get(entity_type)
    if pattern is None:
        return True
    return bool(pattern.match((entity or "").strip()))


# Closed relationship vocabulary
# Must include BOTH the base set AND the functional subset (Layer 6),
# otherwise the contradiction guardrail is dead code.
# ─────────────────────────────────────────────────────────────────────────
ALLOWED_RELATIONSHIPS = frozenset({
    # code-level
    "imports", "calls", "uses", "reads", "writes", "inserts_into",
    "selects_from", "updates", "deletes_from", "depends_on",
    "extends", "implements", "configures",
    # runtime topology
    "connects_to", "validates", "retries", "publishes_to",
    "subscribes_to", "awaits", "emits", "listens_to", "pushes_to",
    "polls", "schedules", "spawns", "owns", "contains",
    # functional / network — must be present, not a separate set
    "runs_on_port", "has_primary_ip", "hosted_in_region",
    # scratchpad-internal
    "has_plan", "involves", "calls_tool", "resolved_by", "blocks",
})

RelationshipType = Literal[
    "imports", "calls", "uses", "reads", "writes", "inserts_into",
    "selects_from", "updates", "deletes_from", "depends_on",
    "extends", "implements", "configures",
    "connects_to", "validates", "retries", "publishes_to",
    "subscribes_to", "awaits", "emits", "listens_to", "pushes_to",
    "polls", "schedules", "spawns", "owns", "contains",
    "runs_on_port", "has_primary_ip", "hosted_in_region",
    "has_plan", "involves", "calls_tool", "resolved_by", "blocks"
]


# Layer 6's contradiction guard only fires for relations where two
# different values for the same (source, relationship) pair is actually
# a contradiction. "A imports B" and "A imports C" is fine (A imports
# both). But "A runs_on_port 5432" and "A runs_on_port 6000" is a
# contradiction.
FUNCTIONAL_RELATIONSHIPS = frozenset({
    "runs_on_port", "has_primary_ip", "hosted_in_region",
})


def validate_relationship(rel: str) -> bool:
    """Layer 1: closed-vocabulary check."""
    return bool(rel) and rel.lower().strip() in ALLOWED_RELATIONSHIPS


# ─────────────────────────────────────────────────────────────────────────
# Direction check: structural, not string-parsed
# ─────────────────────────────────────────────────────────────────────────
# Previously a single free-text field the model was asked to fill in the
# form "[Subject] -> [Verb] -> [Object]", parsed with a regex. Real logs
# showed the model frequently ignoring that convention entirely - values
# like 'rolled out', 'Root cause identified.' with no brackets, no arrows,
# just narrative prose. A str-typed field only constrains the grammar to
# "any valid JSON string"; the bracket format was a description-only
# request the decoder had no way to enforce, and the model treated it as
# optional rather than required.
#
# Three separate required fields close that gap using the SAME mechanism
# that already guarantees the rest of the schema's structure (object with
# required string properties), rather than a regex pattern constraint
# whose support varies significantly across JSON-schema-to-grammar
# backends.

from text_matching import fuzzy_equal


class DirectionCheck(BaseModel):
    """
    In-schema chain-of-thought for direction, as three fields instead of
    one string requiring a specific bracket format. Still generated
    before source_entity/relationship/target_entity (see GraphTriplet
    field order below) - the CoT-forcing property is unchanged, only the
    parsing fragility is removed.
    """
    stated_subject: str = Field(
        ..., description="Who performs the action. Must name the same entity as source_entity below."
    )
    stated_verb: str = Field(..., description="The action or relationship being described.")
    stated_object: str = Field(
        ..., description="Who or what is acted upon. Must name the same entity as target_entity below."
    )


def validate_direction(t: "GraphTriplet") -> bool:
    """
    Second line of defense for the in-schema CoT: the model's stated
    direction must match the fields it filled in. This catches the case
    where the LLM 'thought' one direction but emitted the opposite.

    Uses fuzzy matching rather than exact equality. direction_check and
    source_entity/target_entity are independently-generated fields filled
    in by the same model within one response, and in practice they drift
    in phrasing even when the model got the direction genuinely right -
    "AUTH_SERVICE pods" in one field vs "AUTH_SERVICE" in the other is not
    a direction error, it's the same entity described two ways. A genuine
    subject/object swap still fails this check: swapped entities are not
    fuzzy-similar to each other unless they happen to share very similar
    names, which a real swap error essentially never does.
    """
    return (
        fuzzy_equal(t.direction_check.stated_subject, t.source_entity)
        and fuzzy_equal(t.direction_check.stated_object, t.target_entity)
    )


# ─────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────
class SessionInitRequest(BaseModel):
    session_id: str
    master_plan: Optional[str] = None
    user_query: Optional[str] = None


class GraphTriplet(BaseModel):
    """
    The atomic unit of knowledge in the scratchpad.

    direction_check is an in-schema chain-of-thought field. Pydantic
    emits JSON Schema with properties + required[] in declared field
    order, and the grammar compilers in Ollama / llama.cpp / LM Studio
    generate in that order. By placing direction_check first, the model
    is forced to commit to who-acts-on-whom BEFORE source_entity /
    target_entity are locked in. After parse, we cross-check
    direction_check against the actual fields (validate_direction) —
    this is the second line of defense in case the model skipped the
    grammar-constrained reasoning.
    """
    direction_check: DirectionCheck = Field(
        ...,
        description=(
            "State who acts on whom BEFORE filling the fields below. "
            "stated_subject and stated_object must name the SAME entities "
            "you write in source_entity and target_entity - identical "
            "short names in both places."
        ),
    )
    source_type: EntityType = Field(
        ..., 
        description=(
            "Type of source_entity. MUST be one of the EntityType literals. "
            "CRITICAL: If FILE, entity MUST have a file extension (e.g. config.json). "
            "Do NOT use FILE for components/libraries (use SERVICE). "
            "If PROTOCOL, entity MUST be a standard protocol like http, tcp, etc."
        )
    )
    source_entity: str = Field(
        ...,
        description=(
            "Subject noun, class, or env var in uppercase. Use a SHORT "
            "canonical identifier, e.g. AUTH_SERVICE - not a full phrase "
            "copied from the text, e.g. not 'AUTH SERVICE PODS' or "
            "'THE AUTH SERVICE CONNECTION POOL'. If the fact you're "
            "describing is really about a metric, count, or timestamp "
            "rather than a second named entity, skip this triplet rather "
            "than putting the value in an entity field."
        ),
    )
    relationship: RelationshipType = Field(
        ...,
        description=(
            "Active verb (lowercase_with_underscores). MUST be one of: "
            "imports, calls, uses, reads, writes, inserts_into, selects_from, "
            "updates, deletes_from, depends_on, extends, implements, configures, "
            "connects_to, validates, retries, publishes_to, subscribes_to, awaits, "
            "emits, listens_to, pushes_to, polls, schedules, spawns, owns, contains, "
            "runs_on_port, has_primary_ip, hosted_in_region, has_plan, involves, "
            "calls_tool, resolved_by, blocks. "
            "Any other value will be rejected."
        ),
    )
    target_type: EntityType = Field(
        ..., 
        description=(
            "Type of target_entity. MUST be one of the EntityType literals. "
            "CRITICAL: If FILE, entity MUST have a file extension (e.g. config.json). "
            "Do NOT use FILE for components/libraries (use SERVICE). "
            "If PROTOCOL, entity MUST be a standard protocol like http, tcp, etc."
        )
    )
    target_entity: str = Field(
        ...,
        description=(
            "Object noun, config, or destination in uppercase. Same rule "
            "as source_entity: a short canonical identifier, not a "
            "descriptive phrase, and not a metric/count/timestamp value."
        ),
    )
    citation_quote: str = Field(
        ..., description="Exact substring from raw code validating this connection."
    )


class PageExtractionPayload(BaseModel):
    extracted_triplets: List[GraphTriplet]
    unresolved_variables_mutations: Dict[str, str]
    is_chunk_completely_exhausted: bool


class AgentUpdateRequest(BaseModel):
    agent_id: str
    session_id: str
    raw_active_chunk: str
    extracted_triplets: List[GraphTriplet]
    unresolved_variables_mutations: Dict[str, str]
    is_chunk_completely_exhausted: bool


class MemoryViewResponse(BaseModel):
    session_id: str
    markdown_view: str


class GraphTripletSchema(BaseModel):
    """
    L2 summary triplet. Must mirror GraphTriplet so L2 nodes don't lose
    type information when they're compressed. The sweeper will OVERRIDE
    the model-supplied types with the inherited-from-L1 majority type
    rather than trusting the model to reclassify from scratch.
    """
    source_type: EntityType = Field(
        default="SERVICE",
        description="Inherited from L1 majority type, or SERVICE if mixed.",
    )
    source_entity: str = Field(
        ..., description="The canonical uppercase macro-entity."
    )
    relationship: str = Field(
        ...,
        description=(
            "The connecting verb or system dependency. "
            "MUST be in ALLOWED_RELATIONSHIPS."
        ),
    )
    target_type: EntityType = Field(
        default="SERVICE",
        description="Inherited from L1 majority type, or SERVICE if mixed.",
    )
    target_entity: str = Field(
        ..., description="The target uppercase macro-entity."
    )
    citation_quote: str = Field(
        default="Generated via structural compression.",
        description="System generated reference.",
    )


class L2CompressionPayload(BaseModel):
    reasoning_justification: str = Field(
        ...,
        description="Chain-of-thought explaining the community synthesis.",
    )
    source_edge_ids_used: List[str] = Field(
        ..., description="L1 edge_ids collapsed into this summary."
    )
    extracted_l2_triplets: List[GraphTripletSchema] = Field(
        ..., description="The macro triplets."
    )


class L1ExtractionPayload(BaseModel):
    triplets: List[GraphTriplet]
