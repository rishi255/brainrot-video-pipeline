import json
import os
import shutil
from enum import Enum
from typing import Annotated, Dict, List, Literal, Union

from langchain.memory import ConversationSummaryMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from llm import get_llm
from paths import CONFIG_FILE_PATH, OUTPUTS_DIR, PROMPT_CONFIG_FILE_PATH
from prompt_builder import build_prompt_from_config
from pydantic import BaseModel, BeforeValidator, Field, ValidationError
from TTS.api import TTS
from typing_extensions import TypedDict
from utils import load_config

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


class FactCheckedDialogueOutput(DialogueOutput):
    """Represents a line of dialogue that is checked by the fact checking agent. A line could be approved or not approved."""

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


class VoiceSynthOutput(DialogueOutput):
    """Represents the output of the Voice Synth Node for one dialogue.

    The Voice Synth Node is not an agent, but a simple node that converts the dialogue line to speech using a TTS model.
    """

    audio_path: str = Field(
        description="Path to the generated audio file for the dialogue line."
    )


class VoiceSynthListOutput(BaseModel):
    """Represents the final output of the Voice Synth Node as a list of dialogues.

    The Voice Synth Node is not an agent, but a simple node that converts the dialogue line to speech using a TTS model.
    """

    audio_paths: List[VoiceSynthOutput] = Field(
        description="List of paths to the generated audio files for the dialogue lines."
    )


# ===================
# Define Utility Functions
# ===================


def get_cleaned_topic_name(topic: str) -> str:
    """Cleans the topic name by removing unwanted characters."""
    return "".join(
        char for char in topic if char.isalnum() or char in ("_", " ") or char.isspace()
    ).replace(" ", "_")


def get_topic_directory(topic: str) -> str:
    """Returns the directory path for the given topic."""
    topic_name = get_cleaned_topic_name(topic)
    return os.path.join(OUTPUTS_DIR, topic_name)


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
    voice_synth: VoiceSynthOutput | None  # Output of the Voice Synth Agent
    # video_stitch: VideoStitchOutput | None  # Output of the Video Stitch Agent
    quit: bool  # Whether to exit the bot


# ========== Deterministic Nodes ==========


def starting_node_choice(
    state: AgentState,
) -> Literal["get_user_input", "topic_analyzer"]:
    """Router function to determine which node to start with."""

    state_key_to_nodes = [
        ("topic", "get_user_input"),
        ("topic_analysis", "topic_analyzer"),
        ("research", "research_agent"),
        ("dialogue", "dialogue_generator"),
        ("fact_check", "fact_checker"),
        ("voice_synth", "text_to_speech"),
        # ("video_stitch", "video_stitch"),  # Uncomment if you implement the Video Stitch Agent
        # ("quit", "exit_bot"),  # Uncomment if you want to handle exit in a specific way
    ]

    for state_key, node in state_key_to_nodes:
        # If the state key is not present or its value is None or empty, we start with the corresponding node.
        if state_key not in state or not state[state_key]:
            print(f"🔄 Starting with node named {repr(node)}.")
            return node
        # Even if fact check is populated in state, we also check if it is done and approved.
        elif state_key == "fact_check" and (not is_all_dialogue_approved(state)):
            print(
                "🔄 Starting with node named 'fact_checker' as all dialogue is not approved."
            )
            return node

    # If everything is populated, we can exit the bot.
    return "exit_bot"


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
) -> Literal["dialogue_generator", "text_to_speech"]:
    """Router function to send dialogue(s) for regeneration if not approved. This should be only called after the Fact Checker Agent has run once.
    If all dialogue lines are approved, it proceeds to the Voice Synth Node.
    """
    assert is_fact_checking_done(state)

    if is_all_dialogue_approved(state):
        print("✅ All dialogue lines are approved.")
        return "text_to_speech"

    print(
        "❗ Some dialogue lines need to be regenerated based on fact-checking feedback."
    )
    return "dialogue_generator"


def text_to_speech(state: AgentState) -> dict:
    """Converts the dialogue lines to speech using a text-to-speech model."""
    # Use a TTS model to convert the dialogue lines to audio.
    print("Converting dialogue to speech...")

    tts = TTS("xtts").to("cuda")

    voice_synth_output: list[VoiceSynthOutput] = []

    topic_dir = get_topic_directory(state["topic"])

    audio_dir_for_this_topic = os.path.join(topic_dir, "audio")

    print(f"📂 Output audio directory for this topic: {audio_dir_for_this_topic}")

    # delete output topic directory if it exists
    if os.path.exists(topic_dir):
        print(f"🗑️ Deleting existing topic directory: {topic_dir}")
        shutil.rmtree(topic_dir)

    # (re)create upto the output audio directory for this topic
    os.makedirs(audio_dir_for_this_topic, exist_ok=True)

    for i, dialogue in enumerate(state["fact_check"].dialogue, start=1):
        # at this point dialogue is of type FactCheckedDialogueOutput
        # it is guaranteed to be entirely approved due to the graph definition.
        assert dialogue.approved, "Dialogue line must be approved to synthesize voice."

        audio_path = os.path.join(
            audio_dir_for_this_topic, f"{i}_{dialogue.speaker.value}.mp3"
        )
        tts.tts_to_file(
            text=dialogue.line,
            file_path=audio_path,
            speaker_wav=[SPEAKERS_MAP[dialogue.speaker.value]["audio_path"]],
            language="en",
            split_sentences=True,
        )
        voice_synth_output.append(
            VoiceSynthOutput(
                speaker=dialogue.speaker,
                line=dialogue.line,
                audio_path=audio_path,
            )
        )

    state["voice_synth"] = voice_synth_output
    print("Voice synthesis complete. Audio files generated for each dialogue line.")

    # return the voice_synth_output.
    return {"voice_synth": VoiceSynthListOutput(audio_paths=voice_synth_output)}


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

    input_data = f"Topic: {state['topic']}\nResearch: {state['research']}"

    if is_fact_checking_done(state) and not is_all_dialogue_approved(state):
        # that means dialogue needs to be REgenerated.
        input_data += (
            f"\nNOTE: Some of your generated dialogues were not factually correct. Please fix the ones that are not approved and refer to the review comments provided by the FactChecker."
            f"Fact Checker Output:\n{state['fact_check']}"
        )

    print("Generating dialogue for topic:", state["topic"])

    dialogue_generator_llm = get_llm(model_name=MODEL_NAME, temperature=0)

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
    builder.add_node("text_to_speech", text_to_speech)
    builder.add_node("exit_bot", exit_bot)

    # builder.set_entry_point("get_user_input")
    # Conditional start edge, based on what is defined in the state
    builder.add_conditional_edges(START, starting_node_choice)

    builder.add_conditional_edges("get_user_input", route_choice)

    builder.add_edge("topic_analyzer", "research_agent")
    builder.add_edge("research_agent", "dialogue_generator")
    builder.add_edge("dialogue_generator", "fact_checker")

    # From fact_checker, if all dialogue lines are approved, proceed to text-to-speech.
    builder.add_conditional_edges("fact_checker", feedback_loop_choice)

    builder.add_edge("text_to_speech", "exit_bot")
    builder.add_edge("exit_bot", END)

    return builder.compile()


# ========== Entry Point ==========


def main():
    """Main function to run the video creation pipeline."""
    print("\n🎬 Starting video creation pipeline...")
    graph = build_graph()

    # Get image representation of the graph
    print("\n📊 Generating graph...")
    graph_png_save_path = os.path.join(OUTPUTS_DIR, "flow_graph.png")
    graph.get_graph().draw_mermaid_png(output_file_path=graph_png_save_path)
    print(f"\t💹 Graph image saved as '{graph_png_save_path}'.")

    print("\n🚀 Starting the video creation process...")

    starting_state = {
        "topic": "",
        "topic_analysis": None,
        "research": None,
        "dialogue": None,
        "fact_check": None,
        "voice_synth": None,
        # "video_stitch": None,  # Uncomment if you implement the Video Stitch Agent
        "quit": False,
    }

    final_state: dict = graph.invoke(
        starting_state,
        config={"recursion_limit": 200},
    )
    print("\t✅ Done.")

    final_json = json.dumps(final_state, indent=4, default=lambda o: repr(o))
    json_file_path = os.path.join(
        get_topic_directory(final_state["topic"]), "json_state.json"
    )
    # Save state json to a file
    with open(json_file_path, "w") as f:
        f.write(final_json)
    print(f"\n📄 Final state JSON saved as '{json_file_path}'.")

    final_repr = repr(final_state)
    repr_file_path = os.path.join(
        get_topic_directory(final_state["topic"]), "repr_state.txt"
    )
    # Save state representation to a file
    with open(repr_file_path, "w") as f:
        f.write(final_repr)
    print(f"\n📄 Final state representation saved as '{repr_file_path}'.")

    if "dialogue" in final_state:
        print("Printing Final Dialogue:")
        print(final_state["dialogue"])
    else:
        print("No dialogue was created.")


if __name__ == "__main__":
    main()
