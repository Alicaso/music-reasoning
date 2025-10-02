# Music Theory AI Agent

A sophisticated multi-agent system for automated music theory analysis and question answering using symbolic music notation (Humdrum/Kern format).

## Overview

This project implements an advanced AI agent pipeline that can analyze musical scores and answer complex music theory questions. The system uses a multi-agent architecture with specialized components for planning, reasoning, tool execution, and review.

## Features

- **Multi-Agent Architecture**: Planner → Thinker/Tool-User Loop → Reviewer
- **Comprehensive Music Analysis**: 16 specialized tools for music theory analysis
- **Symbolic Music Processing**: Full support for Humdrum/Kern notation format
- **Advanced Reasoning**: ReAct-style reasoning with tool calling capabilities
- **Multiple Model Support**: Compatible with OpenAI, Claude, Gemini, and other LLMs
- **Structured Validation**: JSON Schema validation for tool parameters
- **Error Handling**: Robust error handling and retry mechanisms

## Architecture

### Core Components

1. **Planner Agent**: Creates strategic analysis plans for music theory questions
2. **Thinker Agent**: Performs reasoning and decides when to use tools
3. **Tool-User Agent**: Executes specific music analysis tools with proper parameters
4. **Reviewer Agent**: Makes final decisions based on all gathered information

### Music Analysis Tools

The system includes 16 specialized tools for music analysis:

- **Pitch Analysis**: Pitch class identification, interval calculation
- **Harmonic Analysis**: Chord progression analysis, Roman numeral analysis
- **Melodic Analysis**: Contour analysis, melodic patterns
- **Rhythmic Analysis**: Duration calculation, tuplet statistics
- **Structural Analysis**: Key estimation, time signature extraction
- **Form Analysis**: Cadence detection, structural statistics

## Installation

### Prerequisites

- Python 3.8+
- Required API keys (OpenAI, Claude, or Gemini)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Alicaso/music-reasoning.git
cd music-reasoning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
# Or for other models:
export ANTHROPIC_API_KEY="your-claude-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

## Usage

### Basic Usage

Run a single question analysis:
```bash
python agent_pipeline_optimized.py --question_id 001
```

### Batch Testing

Test multiple questions:
```bash
# For exam dataset
python test_exam_agent.py --csv examples/test.csv

# For reddit dataset  
python test_reddit_agent.py --csv examples/reddit_data_combined_final.csv
```

### Multi-Model Testing

Compare different models:
```bash
python test_exam_multi_model.py
```

## Configuration

### Environment Variables

- `QUESTIONS_CSV_PATH`: Path to questions CSV file
- `KERN_FOLDER`: Path to Kern notation files
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Claude API key
- `GEMINI_API_KEY`: Gemini API key
- `PLANNER_MODEL`: Model for planner agent (default: gpt-4o-mini)
- `THINKER_MODEL`: Model for thinker agent (default: gpt-4o-mini)
- `TOOLER_MODEL`: Model for tool-user agent (default: gpt-4o-mini)
- `REVIEWER_MODEL`: Model for reviewer agent

### Model Configuration

You can use different models for different components:

```bash
export PLANNER_MODEL="gpt-4o"
export THINKER_MODEL="claude-3-5-sonnet-20241022"
export TOOLER_MODEL="gpt-4o-mini"
export REVIEWER_MODEL="gpt-4o"
```

## Project Structure

```
music-reasoning/
├── README.md
├── requirements.txt
├── .gitignore
├── setup.py
├── agent_pipeline_optimized.py    # Main agent pipeline
├── music21_tools.py               # Music analysis tools
├── music21_tools.yaml             # Tool schemas and descriptions
├── tool_wrapper.py                # Tool output standardization
├── add_schemas.py                 # Schema generation utility
├── test_exam_agent.py             # Exam dataset testing
├── test_exam_react.py             # ReAct pipeline testing
├── test_reddit_agent.py           # Reddit dataset testing
├── test_reddit_react.py           # Reddit ReAct testing
├── test_exam_multi_model.py       # Multi-model comparison
├── data/
│   ├── kern_exam_new/             # Exam dataset Kern files
│   ├── kern_reddit_new/           # Reddit dataset Kern files
│   └── runs/                      # Execution logs
└── examples/
    ├── test.csv                   # Sample exam questions
    └── reddit_data_combined_final.csv # Sample reddit questions
```

## Data Format

### Questions CSV Format

The system expects CSV files with the following columns:
- `question_id`: Unique identifier
- `music_id`: Identifier for the corresponding Kern file
- `question`: The music theory question
- `final_options`: List of answer choices (A, B, C, D)
- `truth_letter`: Correct answer (A, B, C, or D)

### Kern Notation

The system processes musical scores in Humdrum/Kern format, which is a text-based representation of musical notation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [music21](https://web.mit.edu/music21/) for music analysis
- Uses [OpenAI API](https://openai.com/api/) for language models
- Inspired by ReAct reasoning framework
- Thanks to the music theory community for datasets and feedback