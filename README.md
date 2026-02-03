# WebTactix Project

WebTactix is a modular framework for web-based task execution and evaluation. It provides an infrastructure for running automated tasks, evaluating responses, and integrating with different web services using advanced AI models.

## 🧩 Project Structure (Key Files Description)

```

├── webtactix/
│   ├── agents/                # Contains different types of agents for managing tasks
│   │   ├── __init__.py
│   │   ├── constraint_agent.py # Agent responsible for task constraints
│   │   ├── data_agent.py      # Agent responsible for data extraction
│   │   ├── decision_agent.py  # Agent that makes decisions based on the current state
│   │   ├── planner_agent.py   # Agent that plans task execution
│   ├── browser/               # Browser-related modules for web scraping and automation
│   │   ├── __init__.py
│   │   ├── playwright_session.py # Playwright session for browser interaction
│   ├── core/                  # Core modules providing essential utilities and data structures
│   │   ├── __init__.py
│   │   ├── priority_queue.py  # FIFO Queue for managing nodes
│   │   ├── schemas.py         # Data schemas used across the project
│   │   ├── semantic_tree.py   # Semantic tree structure for organizing tasks
│   ├── datasets/              # Modules for handling datasets and task specifications
│   │   ├── __init__.py
│   │   ├── webarena_adapter.py # Adapter for fetching tasks from WebArena repository
│   │   ├── webarena_evaluator.py # Evaluator for assessing task completion
│   ├── llm/                   # Integration with LLM (Large Language Models)
│   │   ├── __init__.py
│   │   ├── openai_compat.py   # Wrapper for OpenAI API
│   │   ├── presets.py         # Presets for different LLM configurations
│   ├── preprocess/            # Modules for preprocessing tasks and observations
│   │   ├── __init__.py
│   │   ├── observation_encoder.py # Encoder for task observations
│   │   ├── snapshot_dedup.py   # Module for deduplication of snapshots
│   ├── runner/                # Handles the execution and running of experiments
│   │   ├── __init__.py
│   │   ├── experiment_runner.py # Runner for experiments
│   │   ├── ft.py              # Additional functionality for the runner
│   │   ├── recorder.py        # Records the results of each task execution
│   ├── tools/                 # Utility tools for the framework
│   ├── workflows/             # Workflows for running predefined sequences of tasks
│   │   ├── __init__.py
│   │   ├── execute.py         # Executes workflows defined by the user
│   │   ├── f_test.py          # Example test workflows
├── main.py                    # Main entry point for running the framework
├── start.sh                   # Shell script to start the experiment
├── README.md                  # This file
└── requirements.txt           # List of Python dependencies
```

## 🔗 Links

- **Introduction**: https://github.com/paper-submission-anoymous/webtactix_introduction  
- **Results**: https://drive.google.com/file/d/1jPKQrfx8dzNP82kBsaW96l-NujUbPDYI/view?usp=drive_link


## 🛠️ Installation

To get started with WebTactix, you can either clone the repository or use it in your existing Python environment. 

### Prerequisites

1. Python 3.7+ (Recommended)
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set the environment variable for `WEBARENA_ROOT`:

```bash
export WEBARENA_ROOT="/path/to/webarena_root"
```

### Dependencies

- **Playwright**: For web scraping and interaction with the browser.
- **OpenAI API**: For interacting with GPT-like models.
- **Other utilities**: Various Python libraries for data manipulation, evaluation, etc.

## 🚀 Running the Experiment

You can run the experiment either through the provided shell script (`start.sh`) or directly using Python.

### Option 1: Using Shell Script

```bash
bash start.sh
```

### Option 2: Using Python Directly

```bash
python main.py
```

## 📝 File Descriptions

### `main.py`
This is the main entry point for running WebTactix. It orchestrates the task execution and evaluation.

### `start.sh`
A shell script for setting up the environment and running the experiment. It simplifies the setup process.

### `webtactix/`
This directory contains all the core logic for WebTactix, divided into several key modules:

- **agents/**: Contains various agents (planner, decision, data) responsible for task planning, decision-making, and data extraction.
- **browser/**: Handles interaction with the web browser using Playwright.
- **core/**: Contains the essential building blocks like the `PriorityQueue` and the `SemanticTree` for organizing tasks.
- **datasets/**: Deals with loading and managing task datasets from WebArena.
- **llm/**: Contains integration code for using large language models (LLMs) such as OpenAI's GPT models.
- **preprocess/**: Preprocessing steps for observations and data.
- **runner/**: Coordinates running the experiments, including tracking the task flow and recording results.
- **tools/**: Miscellaneous utility tools that aid in the execution of tasks.
- **workflows/**: Defines the workflows for running tasks in a predefined order.

### `requirements.txt`
This file lists all the necessary Python libraries and dependencies required to run the project.

## 🧑‍💻 Contributing

We welcome contributions to improve WebTactix. If you want to contribute, feel free to fork the repository and submit a pull request. For any issues or bugs, please create an issue on GitHub.

## 📜 License

WebTactix is licensed under the MIT License. See LICENSE for more details.
