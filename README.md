<p align="center">
  <img src="icon.png" alt="WebTactix Logo" width="900" />
</p>

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

- **Introduction**: https://paper-submission-anoymous.github.io/webtactix_introduction/
- **Results**: https://drive.google.com/file/d/1jPKQrfx8dzNP82kBsaW96l-NujUbPDYI/view?usp=drive_link

## 🛠️ Installation

To get started with WebTactix, you can either clone the repository or use it in your existing Python environment.

### Prerequisites

1. Python 3.7+ (Recommended)
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Running the Code

This project is evaluated on two benchmarks: **WebArena** and **Online Mind2Web**.

### 1) Run on WebArena (Local Deployment)

We deploy the official WebArena services locally (including the map service, see webarena/environment_docker/readme.md for detail). If you also use a local WebArena setup, you can follow our \`reset.sh\` to initialize the WebArena environment, and use \`start_map\` to launch the map service (the URLs and paths inside the script should be replaced with your own configuration).

After the services are up, you can directly run `main.py` to evaluate multiple WebArena tasks in parallel:

```bash
python main.py
```

### 2) Run on Online Mind2Web (Online Website)

Online Mind2Web does not require local services. To switch to the online benchmark, uncomment the following lines in \`main.py\`:

```python
# dataset = "online_mind2web"
# dataset_path = Path("./webtactix/datasets/Online_Mind2Web.json")
# Task_1 = [0, 1, 2, 3, 4, 5]
```

And uncomment the lane task configuration:

```python
# lane_task_ids=[Task_1], # For Online Mind2Web
```

Then, set your `base_url` and API key in `llm/presets.py`, and run:

```bash
python main.py
```

### 3) Visualize Results (Build Record Site)

After evaluation, you can generate an interactive visualization website:

For Online Mind2Web:
```bash
python webtactix/tools/build_record_site.py --base record --dataset online_mind2web --model deepseek
```

For WebArena:
```bash
python webtactix/tools/build_record_site.py --base record --dataset webarena --model deepseek
```

The generated site will be saved to:
- `record/site`

## 🧑‍💻 Contributing

We welcome contributions to improve WebTactix. If you want to contribute, feel free to fork the repository and submit a pull request. For any issues or bugs, please create an issue on GitHub.

## 📜 License

WebTactix is licensed under the MIT License. See LICENSE for more details.
