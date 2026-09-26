# Conversation Graph

The bot is a LangGraph graph that runs **once per user message ("turn")**. Every turn starts at `START`, passes through `classify_professional_content`, and then `route_from_start` picks the entry node based on `last_visited_node` (the last node run in the previous turn). This is how the conversation keeps continuity across turns. `END` means the turn is over and the bot waits for the user's next message.

The graph is split into five diagrams: entry & routing, and one per stage.

## Legend

| Shape / color | Node type |
|---|---|
| 🟨 Hexagon | Classifier (`classify_*`): classifies the user's reply, doesn't write to the user |
| ⬜ Gray rectangle | Fixed message (no LLM) |
| 🟩 Rounded rectangle | LLM-generated text |
| 🟦 Cylinder | Pinecone retrieval (RAG) |
| 🟪 Double rectangle | Internal processing (structured LLM, no user-facing message) |
| 🟧 Diamond | Router `route_from_start` |
| ⬛ | START / END (end of turn: the bot waits for the user's next message) |
| ⬚ Dashed | Reference to a node defined in another diagram |

A labeled edge is conditional: the label is the value returned by the routing function. An unlabeled edge runs immediately, within the same turn.

---

## Stage 0 · Entry & Turn Routing

Every turn starts here. `route_from_start` picks the entry point based on `last_visited_node`. Each block lists the entry nodes of that stage; the text after `←` is the `last_visited_node` value that routes there.

```mermaid
flowchart TD
  START_NODE(["START"])
  END_NODE(["END · end of turn"])
  classify_professional_content{{"classify_professional_content"}}
  route_from_start{"route_from_start"}

  START_NODE --> classify_professional_content --> route_from_start

  subgraph R1["➜ Stage 1 · Opening"]
    direction TB
    ref_classify_opening["classify_opening<br/>← first turn (no opening_status)"]
    ref_classify_content_state["classify_content_state<br/>← invite_to_share / other (opening_status set)"]
    ref_classify_respond_check_choice["classify_respond_check_choice<br/>← respond_with_check"]
    ref_classify_direction_choice["classify_direction_choice<br/>← ask_direction"]
    ref_classify_practical_check_choice["classify_practical_check_choice<br/>← ask_direction_practical_check"]
    ref_classify_opening ~~~ ref_classify_content_state ~~~ ref_classify_respond_check_choice ~~~ ref_classify_direction_choice ~~~ ref_classify_practical_check_choice
  end

  subgraph R2["➜ Stage 2 · Emotional Track"]
    direction TB
    ref_classify_present_choice["classify_present_choice<br/>← present_and_ask (blocks exist)"]
    ref_classify_block_target["classify_block_target<br/>← ask_which_block / classify_block_target"]
    ref_deepen_round["deepen_round<br/>← deepen_reply"]
    ref_classify_focus_choice["classify_focus_choice<br/>← classify_focus_choice / focus_on_block (no block chosen by user)"]
    ref_focus_on_block["focus_on_block<br/>← unreachable (listed as a target, never returned)"]
    ref_classify_present_choice ~~~ ref_classify_block_target ~~~ ref_deepen_round ~~~ ref_classify_focus_choice ~~~ ref_focus_on_block
  end

  subgraph R3["➜ Stage 3 · Success Moments Analysis"]
    direction TB
    ref_classify_success_consent["classify_success_consent<br/>← present_success_analysis_intro"]
    ref_classify_scope_creep["classify_scope_creep<br/>← invite_success_story / success_analysis_conversation"]
    ref_classify_pivot_consent["classify_pivot_consent<br/>← pivot_to_deeper_process"]
    ref_classify_success_consent ~~~ ref_classify_scope_creep ~~~ ref_classify_pivot_consent
  end

  subgraph R4["➜ Stage 4 · Practical Track"]
    direction TB
    ref_classify_practical_track_consent["classify_practical_track_consent<br/>← present_practical_track_intro"]
    ref_classify_from_practical_track["classify_from_practical_track<br/>← practical_track_conversation (otherwise)"]
    ref_classify_practical_to_success_consent["classify_practical_to_success_consent<br/>← pivot_practical_to_success"]
    ref_classify_practical_track_consent ~~~ ref_classify_from_practical_track ~~~ ref_classify_practical_to_success_consent
  end

  route_from_start --> R1
  route_from_start --> R2
  route_from_start --> R3
  route_from_start --> R4
  route_from_start -->|"practical_track_conversation + #quot;יצאתי לחשוב#quot;"| END_NODE

  classDef classify fill:#fff3cd,stroke:#b8860b,color:#000
  classDef router fill:#ffe5d0,stroke:#fd7e14,color:#000
  classDef terminal fill:#343a40,stroke:#000,color:#fff
  classDef ref fill:#fff,stroke:#999,stroke-dasharray:4 3,color:#555
  classDef unreachable fill:#fff,stroke:#ccc,stroke-dasharray:2 4,color:#aaa

  class classify_professional_content classify
  class route_from_start router
  class START_NODE,END_NODE terminal
  class ref_classify_opening,ref_classify_content_state,ref_classify_respond_check_choice,ref_classify_direction_choice,ref_classify_practical_check_choice,ref_classify_present_choice,ref_classify_block_target,ref_deepen_round,ref_classify_focus_choice,ref_classify_success_consent,ref_classify_scope_creep,ref_classify_pivot_consent,ref_classify_practical_track_consent,ref_classify_from_practical_track,ref_classify_practical_to_success_consent ref
  class ref_focus_on_block unreachable
```

`classify_professional_content` runs on every turn. If the content is professional, it adds a disclaimer message and continues. `"יצאתי לחשוב"` means "I'm off to think".

---

## Stage 1 · Opening

```mermaid
flowchart TD
  from_route(["↩ route_from_start"])
  END_NODE(["END · end of turn"])

  classify_opening{{"classify_opening"}}
  respond_direct["respond_direct"]
  respond_with_check("respond_with_check")
  classify_respond_check_choice{{"classify_respond_check_choice"}}
  invite_to_share("invite_to_share")
  classify_content_state{{"classify_content_state"}}
  ask_direction("ask_direction")
  classify_direction_choice{{"classify_direction_choice"}}
  classify_practical_check_choice{{"classify_practical_check_choice"}}

  ref_retrieve_expressions_content["➜ retrieve_expressions_content · Stage 2"]
  ref_present_practical_track_intro["➜ present_practical_track_intro · Stage 4"]

  from_route -->|"first turn"| classify_opening
  from_route -->|"respond_with_check"| classify_respond_check_choice
  from_route -->|"invite_to_share / other + opening_status set"| classify_content_state
  from_route -->|"ask_direction"| classify_direction_choice
  from_route -->|"ask_direction_practical_check"| classify_practical_check_choice

  classify_opening -->|"opening_status ∈ 1,3,5"| respond_direct
  classify_opening -->|"opening_status ∈ 2,4"| respond_with_check
  respond_direct -->|"end (opening_status=1)"| END_NODE
  respond_direct -->|"continue"| classify_content_state
  classify_respond_check_choice -->|"yes"| invite_to_share
  classify_respond_check_choice -->|"no [end]"| END_NODE
  classify_respond_check_choice -->|"other"| classify_content_state
  classify_content_state -->|"emotional_vague / dual / practical_clear"| ask_direction
  classify_content_state -->|"emotional_clear"| ref_retrieve_expressions_content
  classify_content_state -->|"other [end]"| END_NODE
  classify_direction_choice -->|"pause"| invite_to_share
  classify_direction_choice -->|"continue"| ref_present_practical_track_intro
  classify_practical_check_choice -->|"no"| ref_present_practical_track_intro
  classify_practical_check_choice -->|"yes / other"| classify_content_state

  respond_with_check & invite_to_share & ask_direction --> END_NODE

  classDef classify fill:#fff3cd,stroke:#b8860b,color:#000
  classDef fixed fill:#e2e3e5,stroke:#6c757d,color:#000
  classDef llm fill:#d1e7dd,stroke:#198754,color:#000
  classDef terminal fill:#343a40,stroke:#000,color:#fff
  classDef ref fill:#fff,stroke:#999,stroke-dasharray:4 3,color:#555

  class classify_opening,classify_respond_check_choice,classify_content_state,classify_direction_choice,classify_practical_check_choice classify
  class respond_direct fixed
  class respond_with_check,invite_to_share,ask_direction llm
  class END_NODE terminal
  class from_route,ref_retrieve_expressions_content,ref_present_practical_track_intro ref
```

---

## Stage 2 · Emotional Track: Expressions, Blocks, Deepening

```mermaid
flowchart TD
  from_route(["↩ route_from_start"])
  END_NODE(["END · end of turn"])
  ref_classify_content_state["Stage 1 · classify_content_state ➜"]
  ref_classify_pivot_consent["Stage 3 · classify_pivot_consent ➜"]

  retrieve_expressions_content[("retrieve_expressions_content")]
  build_expressions_table[["build_expressions_table"]]
  build_blocks[["build_blocks"]]
  color_blocks[["color_blocks"]]
  present_and_ask("present_and_ask")
  classify_present_choice{{"classify_present_choice"}}
  ask_which_block("ask_which_block")
  classify_block_target{{"classify_block_target"}}
  deepen_round[["deepen_round"]]
  deepen_reply("deepen_reply")
  focus_on_block("focus_on_block")
  classify_focus_choice{{"classify_focus_choice"}}
  classify_readiness{{"classify_readiness"}}
  summarize_and_pivot("summarize_and_pivot")

  ref_classify_content_state -->|"emotional_clear"| retrieve_expressions_content
  ref_classify_pivot_consent -->|"yes"| retrieve_expressions_content

  from_route -->|"present_and_ask (blocks exist)"| classify_present_choice
  from_route -->|"ask_which_block / classify_block_target"| classify_block_target
  from_route -->|"deepen_reply"| deepen_round
  from_route -->|"classify_focus_choice / focus_on_block (no block chosen by user)"| classify_focus_choice
  from_route -.->|"unreachable"| focus_on_block

  retrieve_expressions_content --> build_expressions_table --> build_blocks --> color_blocks --> present_and_ask

  classify_present_choice -->|"intent=deepen, no block given"| ask_which_block
  classify_present_choice -->|"otherwise [end]"| END_NODE
  classify_block_target -->|"valid block & count #lt; 2"| deepen_round
  classify_block_target -->|"otherwise [end]"| END_NODE
  deepen_round -->|"count #lt; 2"| deepen_reply
  deepen_round -->|"count ≥ 2"| focus_on_block
  classify_focus_choice -->|"last message is bot's (repeat request) [end]"| END_NODE
  classify_focus_choice -->|"otherwise"| classify_readiness
  classify_readiness -->|"not_ready"| summarize_and_pivot
  classify_readiness -->|"otherwise [end]"| END_NODE

  present_and_ask & ask_which_block & deepen_reply & focus_on_block & summarize_and_pivot --> END_NODE

  classDef classify fill:#fff3cd,stroke:#b8860b,color:#000
  classDef llm fill:#d1e7dd,stroke:#198754,color:#000
  classDef rag fill:#cfe2ff,stroke:#0d6efd,color:#000
  classDef internal fill:#e2d9f3,stroke:#6f42c1,color:#000
  classDef terminal fill:#343a40,stroke:#000,color:#fff
  classDef ref fill:#fff,stroke:#999,stroke-dasharray:4 3,color:#555

  class classify_present_choice,classify_block_target,classify_focus_choice,classify_readiness classify
  class present_and_ask,ask_which_block,deepen_reply,focus_on_block,summarize_and_pivot llm
  class retrieve_expressions_content rag
  class build_expressions_table,build_blocks,color_blocks,deepen_round internal
  class END_NODE terminal
  class from_route,ref_classify_content_state,ref_classify_pivot_consent ref
```

`count` refers to `deepen_round_count`.

---

## Stage 3 · Success Moments Analysis

```mermaid
flowchart TD
  from_route(["↩ route_from_start"])
  END_NODE(["END · end of turn"])
  ref_classify_practical_to_success_consent["Stage 4 · classify_practical_to_success_consent ➜"]

  present_success_analysis_intro["present_success_analysis_intro"]
  classify_success_consent{{"classify_success_consent"}}
  explain_success_value("explain_success_value")
  invite_success_story["invite_success_story"]
  classify_scope_creep{{"classify_scope_creep"}}
  pivot_to_deeper_process["pivot_to_deeper_process"]
  classify_pivot_consent{{"classify_pivot_consent"}}
  retrieve_success_analysis_context[("retrieve_success_analysis_context")]
  success_analysis_conversation("success_analysis_conversation")

  ref_retrieve_expressions_content["➜ retrieve_expressions_content · Stage 2"]

  ref_classify_practical_to_success_consent -->|"yes"| present_success_analysis_intro

  from_route -->|"present_success_analysis_intro"| classify_success_consent
  from_route -->|"invite_success_story / success_analysis_conversation"| classify_scope_creep
  from_route -->|"pivot_to_deeper_process"| classify_pivot_consent

  classify_success_consent -->|"yes"| invite_success_story
  classify_success_consent -->|"no"| explain_success_value
  classify_success_consent -->|"other"| classify_scope_creep
  classify_scope_creep -->|"drifting"| pivot_to_deeper_process
  classify_scope_creep -->|"in_scope"| retrieve_success_analysis_context
  classify_pivot_consent -->|"yes"| ref_retrieve_expressions_content
  classify_pivot_consent -->|"no / other"| retrieve_success_analysis_context
  retrieve_success_analysis_context --> success_analysis_conversation

  present_success_analysis_intro & explain_success_value & invite_success_story & pivot_to_deeper_process & success_analysis_conversation --> END_NODE

  classDef classify fill:#fff3cd,stroke:#b8860b,color:#000
  classDef fixed fill:#e2e3e5,stroke:#6c757d,color:#000
  classDef llm fill:#d1e7dd,stroke:#198754,color:#000
  classDef rag fill:#cfe2ff,stroke:#0d6efd,color:#000
  classDef terminal fill:#343a40,stroke:#000,color:#fff
  classDef ref fill:#fff,stroke:#999,stroke-dasharray:4 3,color:#555

  class classify_success_consent,classify_scope_creep,classify_pivot_consent classify
  class present_success_analysis_intro,invite_success_story,pivot_to_deeper_process fixed
  class explain_success_value,success_analysis_conversation llm
  class retrieve_success_analysis_context rag
  class END_NODE terminal
  class from_route,ref_classify_practical_to_success_consent,ref_retrieve_expressions_content ref
```

---

## Stage 4 · Practical Track

```mermaid
flowchart TD
  from_route(["↩ route_from_start"])
  END_NODE(["END · end of turn"])
  ref_classify_direction_choice["Stage 1 · classify_direction_choice ➜"]
  ref_classify_practical_check_choice["Stage 1 · classify_practical_check_choice ➜"]

  present_practical_track_intro["present_practical_track_intro"]
  classify_practical_track_consent{{"classify_practical_track_consent"}}
  practical_track_conversation("practical_track_conversation")
  classify_from_practical_track{{"classify_from_practical_track"}}
  pivot_practical_to_success["pivot_practical_to_success"]
  classify_practical_to_success_consent{{"classify_practical_to_success_consent"}}

  ref_present_success_analysis_intro["➜ present_success_analysis_intro · Stage 3"]

  ref_classify_direction_choice -->|"continue"| present_practical_track_intro
  ref_classify_practical_check_choice -->|"no"| present_practical_track_intro

  from_route -->|"present_practical_track_intro"| classify_practical_track_consent
  from_route -->|"practical_track_conversation (otherwise)"| classify_from_practical_track
  from_route -->|"practical_track_conversation + exact text #quot;יצאתי לחשוב#quot;"| END_NODE
  from_route -->|"pivot_practical_to_success"| classify_practical_to_success_consent

  classify_practical_track_consent -->|"no [end]"| END_NODE
  classify_practical_track_consent -->|"yes / other"| practical_track_conversation
  classify_from_practical_track -->|"pausing [end]"| END_NODE
  classify_from_practical_track -->|"capability_doubt"| pivot_practical_to_success
  classify_from_practical_track -->|"continuing"| practical_track_conversation
  classify_practical_to_success_consent -->|"yes"| ref_present_success_analysis_intro
  classify_practical_to_success_consent -->|"no / other"| practical_track_conversation

  present_practical_track_intro & practical_track_conversation & pivot_practical_to_success --> END_NODE

  classDef classify fill:#fff3cd,stroke:#b8860b,color:#000
  classDef fixed fill:#e2e3e5,stroke:#6c757d,color:#000
  classDef llm fill:#d1e7dd,stroke:#198754,color:#000
  classDef terminal fill:#343a40,stroke:#000,color:#fff
  classDef ref fill:#fff,stroke:#999,stroke-dasharray:4 3,color:#555

  class classify_practical_track_consent,classify_from_practical_track,classify_practical_to_success_consent classify
  class present_practical_track_intro,pivot_practical_to_success fixed
  class practical_track_conversation llm
  class END_NODE terminal
  class from_route,ref_classify_direction_choice,ref_classify_practical_check_choice,ref_present_success_analysis_intro ref
```
