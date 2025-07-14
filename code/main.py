import json
from typing import List, Literal

from enum import Enum
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from llm import get_llm
from paths import CONFIG_FILE_PATH, PROMPT_CONFIG_FILE_PATH
from prompt_builder import build_prompt_from_config
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from utils import load_config
from pydantic import BaseModel, Field, ValidationError, BeforeValidator
from typing import Literal, Union, Dict, Annotated

# ========== Prompt Config ==========

prompt_cfg: dict = load_config(PROMPT_CONFIG_FILE_PATH)
app_cfg: dict = load_config(CONFIG_FILE_PATH)

MODEL_NAME = app_cfg["llm"]

# list of speakers.
# For each speaker:
# <speaker_name>: {audio_path: str, image_path: str}
SPEAKERS_MAP = app_cfg["speakers"]
# [{'Saul Goodman': {'audio_path': 'assets/audio/saul.wav', 'image_path': 'assets/images/saul.png'}}, {'Walter White': {'audio_path': 'assets/audio/walter.wav', 'image_path': 'assets/images/walter.png'}}]

# Convert this into a single dictionary for easier access
SPEAKERS_MAP = {
    speaker_name: {
        "audio_path": speaker_info["audio_path"],
        "image_path": speaker_info["image_path"],
    }
    for speaker in SPEAKERS_MAP
    for speaker_name, speaker_info in speaker.items()
}

# Now it's of the form:
# {'Saul Goodman': {'audio_path': 'assets/audio/saul.wav', 'image_path': 'assets/images/saul.png'}, 'Walter White': {'audio_path': 'assets/audio/walter.wav', 'image_path': 'assets/images/walter.png'}}

ValidSpeakers = Enum(  # type: ignore[misc]
    "ValidSpeakers",
    ((value, value) for value in SPEAKERS_MAP.keys()),
    type=str,
)

# print("Valid Speakers Enum:")
# print(ValidSpeakers.__members__.keys())

# ===================
# Define Structured Output Classes for Agents
# ===================


class TopicAnalysisOutput(BaseModel):
    """Represents the output of the Topic Analyzer Agent."""

    category: str = Field(description="The category of the input topic.")
    subtopics: List[str] = Field(
        default_factory=list,
        description="List of important subtopics related to the input topic, that can be covered in the educational video.",
    )

    # def __repr__(self):
    #     return f"TopicAnalysisOutput(category={self.category}, subtopics={self.subtopics})"


class ResearchOutput(BaseModel):
    """Represents the output of the Research Agent."""

    facts: List[str] = Field(
        description="List of factual points extracted from research to cover in the educational video."
    )


class Speaker(BaseModel):
    """Represents a speaker in the dialogue."""

    name: str = Field(description="The name of the speaker.")
    sample_audio_path: str = Field(
        description="Path to a sample audio file for voice cloning."
    )
    image_path: str = Field(description="Path to an image of the speaker.")


class DialogueOutput(BaseModel):
    """Represents a line of dialogue."""

    speaker: ValidSpeakers = Field(description="The speaker of the line.")
    line: str = Field(description="The dialogue line.")


class FactCheckedDialogueOutput(BaseModel):
    """Represents a line of dialogue that is checked by the fact checking agent. A line could be approved or not approved."""

    speaker: ValidSpeakers = Field(description="The speaker of the line.")
    line: str = Field(description="The dialogue line.")
    approved: bool = Field(
        default=False,
        description="Whether the line has been fact-checked and approved by the FactChecker Agent.",
    )
    review_comments: str = Field(
        default="",
        description="Review comments from the FactChecker Agent on the dialogue line if not approved. This should be used to regenerate the dialogue line.",
    )


class DialogueListOutput(BaseModel):
    """Represents the output of the Dialogue Generator Agent."""

    dialogue: List[DialogueOutput] = Field(
        description="List of dialogue lines for the video."
    )


class FactCheckedDialogueListOutput(BaseModel):
    """Represents the output of the Fact Checker Agent."""

    dialogue: List[FactCheckedDialogueOutput] = Field(
        description="List of fact-checked dialogue lines for the video."
    )


# ===================
# Define State
# ===================


class AgentState(TypedDict):
    """State for the agentic video creation process."""

    topic: str  # The user input topic
    topic_analysis: TopicAnalysisOutput | None  # Output of the Topic Analyzer Agent
    research: ResearchOutput | None  # Output of the Research Agent
    dialogue: DialogueListOutput | None  # Output of the Dialogue Generator Agent
    fact_check: FactCheckedDialogueListOutput | None  # Output of the Fact Checker Agent
    # voice_synth: VoiceSynthOutput | None  # Output of the Voice Synth Agent
    # video_stitch: VideoStitchOutput | None  # Output of the Video Stitch Agent
    quit: bool  # Whether to exit the bot


# ========== Deterministic Nodes ==========


def get_user_input(state: AgentState) -> dict:
    """Gets user input for the topic of the video."""
    topic = input("Enter the topic for the video (or type 'exit' to quit): ").strip()
    return {"topic": topic}


def route_choice(
    state: AgentState,
) -> Literal["get_user_input", "topic_analyzer", "exit_bot"]:
    """Router function to determine whether to exit or proceed."""
    if not state["topic"]:
        print("❌ No input provided. Please enter a topic.")
        return "get_user_input"
    elif state["topic"].lower() == "exit":
        return "exit_bot"
    else:
        return "topic_analyzer"


def is_fact_checking_done(state: AgentState) -> bool:
    """Checks if the fact-checking output is valid."""
    if "fact_check" not in state:
        return False
    if not isinstance(state["fact_check"], FactCheckedDialogueListOutput):
        return False
    return True


def is_all_dialogue_approved(state: AgentState) -> bool:
    """Checks if all dialogue lines are approved."""
    if not is_fact_checking_done(state):
        return False
    for line in state["fact_check"].dialogue:
        if not line.approved:
            return False
    return True


def feedback_loop_choice(
    state: AgentState,
) -> Literal["dialogue_generator", "exit_bot"]:
    """Router function to send dialogue(s) for regeneration if not approved. This should be only called after the Fact Checker Agent has run once."""
    assert is_fact_checking_done(state)

    if is_all_dialogue_approved(state):
        print("✅ All dialogue lines are approved.")
        return "exit_bot"

    print(
        "❗ Some dialogue lines need to be regenerated based on fact-checking feedback."
    )
    return "dialogue_generator"


def exit_bot(state: AgentState) -> dict:
    """Exits the bot."""
    print("\n" + "🚪" + "=" * 58 + "🚪")
    print("    GOODBYE!")
    print("=" * 60)
    return {"quit": True}


# ========== LLM Nodes ==========


def topic_analyzer(state: AgentState) -> dict:
    """Analyzes the topic and extracts subtopics."""

    print("Analyzing topic:", state["topic"])

    # TODO: Include web search results also in the prompt for the topic analysis.

    topic_analyzer_llm = get_llm(model_name=MODEL_NAME, temperature=0)
    prompt = build_prompt_from_config(
        prompt_cfg["TopicAnalyzer_prompt_cfg"], input_data=state["topic"]
    )
    response: TopicAnalysisOutput = topic_analyzer_llm.with_structured_output(
        TopicAnalysisOutput
    ).invoke(prompt)
    return {"topic_analysis": response}


def research_agent(state: AgentState) -> dict:
    """Researches the topic and extracts facts."""

    print("Researching topic:", state["topic"])

    # TODO: Include web search results also in the prompt for each subtopic.

    research_agent_llm = get_llm(model_name=MODEL_NAME, temperature=0)
    prompt = build_prompt_from_config(
        prompt_cfg["ResearchAgent_prompt_cfg"],
        input_data=f"Topic: {state['topic']}\nTopic Analysis: {state['topic_analysis']}",
    )
    response: ResearchOutput = research_agent_llm.with_structured_output(
        ResearchOutput
    ).invoke(prompt)

    return {"research": response}


def dialogue_generator(state: AgentState) -> dict:
    """Generates dialogue based on the research."""

    print("Generating dialogue for topic:", state["topic"])

    dialogue_generator_llm = get_llm(model_name=MODEL_NAME, temperature=0)

    input_data = f"Topic: {state['topic']}\nResearch: {state['research']}"

    if is_fact_checking_done(state):
        # that means dialogue needs to be REgenerated.
        input_data += (
            f"\nNOTE: Some of your generated dialogues were not factually correct. Please fix the ones that are not approved and refer to the review comments provided by the FactChecker."
            + f"Fact Checker Output:\n{state['fact_check']}"
        )

    prompt = build_prompt_from_config(
        prompt_cfg["DialogueGenerator_prompt_cfg"],
        input_data=input_data,
    )

    response: DialogueListOutput = dialogue_generator_llm.with_structured_output(
        DialogueListOutput
    ).invoke(prompt)

    return {"dialogue": response}


def fact_checker(state: AgentState) -> dict:
    """Checks the facts in the dialogue lines and approves them if they are correct."""
    print("Fact-checking dialogue for topic:", state["topic"])
    fact_checker_llm = get_llm(model_name=MODEL_NAME, temperature=0)
    prompt = build_prompt_from_config(
        prompt_cfg["FactChecker_prompt_cfg"],
        input_data=f"Topic: {state['topic']}\nDialogue: {state['dialogue']}",
    )
    response: FactCheckedDialogueListOutput = fact_checker_llm.with_structured_output(
        FactCheckedDialogueListOutput
    ).invoke(prompt)

    return {"fact_check": response}


# ========== Tool Definitions ==========


def get_web_search_tool():
    return TavilySearch(max_results=5, search_depth="advanced")


def get_tools():
    return [get_web_search_tool()]


# ========== Agent Node Factories ==========
# (No agent node factories needed for this simplified version)


# ========== Graph Assembly ==========


def build_graph() -> CompiledStateGraph:
    """Builds the LangGraph graph."""

    builder = StateGraph(AgentState)

    builder.add_node("get_user_input", get_user_input)
    builder.add_node("topic_analyzer", topic_analyzer)
    builder.add_node("research_agent", research_agent)
    builder.add_node("dialogue_generator", dialogue_generator)
    builder.add_node("fact_checker", fact_checker)
    builder.add_node("exit_bot", exit_bot)

    builder.set_entry_point("get_user_input")

    builder.add_conditional_edges("get_user_input", route_choice)
    builder.add_conditional_edges("fact_checker", feedback_loop_choice)

    builder.add_edge("topic_analyzer", "research_agent")
    builder.add_edge("research_agent", "dialogue_generator")
    builder.add_edge("dialogue_generator", "fact_checker")
    builder.add_edge("fact_checker", "exit_bot")
    builder.add_edge("exit_bot", END)

    return builder.compile()


# ========== Entry Point ==========


def main():
    """Main function to run the video creation pipeline."""
    print("\n🎬 Starting video creation pipeline...")
    graph = build_graph()

    # Get image representation of the graph
    print("\n📊 Generating graph...")
    graph.get_graph().draw_mermaid_png(output_file_path="video_creation_graph.png")
    print("\t💹 Graph image saved as 'video_creation_graph.png'.")

    print("\n🚀 Starting the video creation process...")
    final_state: dict = graph.invoke(
        {
            "topic": "",
            "topic_analysis": None,
            "research": None,
            "dialogue": None,
            "quit": False,
        },
        config={"recursion_limit": 200},
    )
    print("\t✅ Done.")

    # Convert the final_state dictionary to a JSON string
    final_state_json = json.dumps(final_state, indent=4, default=lambda o: o.__dict__)
    # Save final state json to a file
    with open("final_video_state.json", "w") as f:
        f.write(final_state_json)

    print("\n📄 Final state saved as 'final_video_state.json'.")

    if "dialogue" in final_state:
        print("Printing Final Dialogue:")
        print(final_state["dialogue"])
    else:
        print("No dialogue was created.")


if __name__ == "__main__":
    main()
