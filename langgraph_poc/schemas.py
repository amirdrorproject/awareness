from typing import Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from langgraph_poc.common import PROMPTS


class OpeningClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the chosen mode")
    mode: int = Field(description="The opening mode, 1 to 5", ge=1, le=5)


class ContentStateClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the chosen state")
    state: Literal["emotional_clear", "emotional_vague", "practical_clear", "dual"]


class DirectionChoiceClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the chosen direction")
    choice: Literal["pause", "continue"]


class YesNoOtherResult(BaseModel):
    reasoning: str
    answer: Literal["yes", "no", "other"]


class TableRow(BaseModel):
    row_number: int
    expression: str
    expression_type: Literal["גלוי", "מרומז", "פיזי"]
    bank_name: str
    matched_expression: str
    match_level: Literal["זהה", "דומה", "קרוב", "מנוגד"]
    block_id: int | None = None


class ExpressionsTableResult(BaseModel):
    reasoning: str
    rows: list[TableRow]


class Block(BaseModel):
    block_id: int
    topic: str
    color: str | None = None


class BuildBlocksResult(BaseModel):
    reasoning: str
    blocks: list[Block]
    rows: list[TableRow]


class BlockColorAssignment(BaseModel):
    block_id: int
    color: str


class ColorBlocksResult(BaseModel):
    reasoning: str
    colors: list[BlockColorAssignment]


class PresentChoiceResult(BaseModel):
    reasoning: str
    intent: Literal["practical", "deepen"]
    current_block_id: int | None = None


class BlockTargetResult(BaseModel):
    reasoning: str
    current_block_id: int


class DeepenRoundResult(BaseModel):
    reasoning: str
    new_rows: list[TableRow]
    block_updates: list[Block]


class FocusChoiceResult(BaseModel):
    reasoning: str
    current_block_id: int


class ReadinessResult(BaseModel):
    reasoning: str
    readiness: Literal["ready", "half_ready", "not_ready"]


class ScopeCreepResult(BaseModel):
    reasoning: str
    status: Literal["in_scope", "drifting"]


class FromPracticalTrackResult(BaseModel):
    reasoning: str
    status: Literal["pausing", "capability_doubt", "continuing"]


class ReflectionContext(BaseModel):
    reflection_stage_1: str = Field(description=PROMPTS["reflection_stage_1"])
    reflection_stage_2: str = Field(description=PROMPTS["reflection_stage_2"])
    reflection_stage_3: str = Field(description=PROMPTS["reflection_stage_3"])
    reflection_stage_4: str = Field(description=PROMPTS["reflection_stage_4"])


class ProfessionalContentClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the classification")
    is_professional: bool


class GraphState(MessagesState):
    internal_audit_log: str
    opening_status: int
    content_state: str
    direction_choice: str
    expressions_content: list[dict]
    expressions_table: list[dict]
    blocks: list[dict]
    intent: str
    current_block_id: int | None
    block_chosen_unprompted: bool
    deepen_round_count: int
    deepen_round_new_rows: list[dict]
    deepen_round_block_updates: list[dict]
    readiness: str
    respond_check_answer: Literal["yes", "no", "other"] | None
    practical_check_answer: Literal["yes", "no", "other"] | None
    success_analysis_start_index: int | None
    success_analysis_rag_context: list[dict]
    success_analysis_rag_query: str | None
    success_consent_answer: Literal["yes", "no", "other"] | None
    scope_creep_status: Literal["in_scope", "drifting"] | None
    pivot_consent_answer: Literal["yes", "no", "other"] | None
    practical_track_consent_answer: Literal["yes", "no", "other"] | None
    practical_track_pause_status: Literal["pausing", "capability_doubt", "continuing"] | None
    practical_to_success_consent_answer: Literal["yes", "no", "other"] | None
    last_visited_node: str | None
