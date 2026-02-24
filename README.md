<p align="center">
  <img src="icon.png" alt="WebTactix Logo" width="900" />
</p>

# WebTactix Project

WebTactix is a modular framework for web-based task execution and evaluation. It provides an infrastructure for running automated tasks, evaluating responses, and integrating with different web services using advanced AI models.

## 🧩 Project Structure (Key Files Description)

```
├── webtactix/
│   ├── agents/                  # Agents for planning/decision/constraints/data extraction
│   │   ├── __init__.py
│   │   ├── constraint_agent.py  # Extracts/maintains task constraints
│   │   ├── data_agent.py        # Data extraction agent (e.g., collecting target info)
│   │   ├── decision_agent.py    # Chooses next branch/action based on current state
│   │   └── planner_agent.py     # Generates candidate plans/actions
│   ├── browser/                 # Browser automation layer
│   │   ├── __init__.py
│   │   └── playwright_session.py# Playwright session wrapper
│   ├── core/                    # Core data structures and utilities
│   │   ├── __init__.py
│   │   ├── priority_queue.py    # Priority queue / frontier management
│   │   ├── schemas.py           # Shared data schemas
│   │   └── semantic_tree.py     # Semantic tree memory structure
│   ├── datasets/                # Dataset adapters and evaluators
│   │   ├── __init__.py
│   │   ├── online_min2web_adapter.py # Adapter for Online Mind2Web tasks
│   │   ├── Online Mind2Web.json      # Example/config for Online Mind2Web
│   │   ├── test_evaluator.py         # Lightweight evaluator for testing
│   │   ├── webarena_adapter.py       # Adapter for WebArena tasks
│   │   └── webarena_evaluator.py     # WebArena evaluator
│   ├── llm/                     # LLM interface and presets
│   │   ├── __init__.py
│   │   ├── openai_compat.py     # OpenAI-compatible API wrapper
│   │   └── presets.py           # Presets/configs for different LLM setups
│   ├── preprocess/              # Observation preprocessing and dedup
│   │   ├── __init__.py
│   │   ├── observation_encoder.py # Encodes observations (e.g., AxTree processing)
│   │   └── snapshot_dedup.py      # Snapshot deduplication
│   ├── runner/                  # Experiment execution orchestration
│   │   ├── __init__.py
│   │   ├── experiment_runner.py  # Main experiment runner
│   │   ├── ft.py                 # Use for environment test (can ignore)
│   │   └── recorder.py           # Recording/logging of trajectories/results
│   └── workflows/               # Workflow entrypoints
│       ├── __init__.py
│       └── execute.py            # Executes a workflow / task loop
├── tools/                       # Utility scripts (record visualization / inspection)
│   ├── __init__.py
│   ├── build_record_site.py      # Build visualization site from records
│   └── inspect_record.py         # Inspect records locally
├── main.py                       # Main entry point
├── start.sh                      # Convenience script to start webarena environment
├── README.md
└── requirements.txt
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

We deploy the official WebArena services locally (including the map service, see webarena/environment_docker/readme.md for detail). If you also use a local WebArena setup, you can follow our `reset.sh` to initialize the WebArena environment, and use `start_map.sh` to launch the map service (the URLs and paths inside the script should be replaced with your own configuration).

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
