# Hermes: Efficient Serving of LLM Applications with Probabilistic Demand Modeling

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Paper](https://doi.org/10.1145/3803390) | [arXiv](https://arxiv.org/abs/2506.14851) | [Project structure](#project-structure) | [Installation](#installation) | [Running Hermes](#running-hermes) | [Citation](#citation)

## About

Hermes is a serving system for compound LLM applications whose requests form multi-stage workflows with dynamic demand on different backends. It models each application with a Probabilistic Demand Graph (PDGraph), uses a Gittins policy to schedule applications, and prewarms backends based on predicted future demand.

The repository combines the CTaskBench application benchmark with a modified vLLM 0.4.3 serving stack. In the reported experiments, Hermes reduces average application completion time by more than 70% and P95 completion time by more than 80%.

## Project structure

```text
.
├── CTaskBench/
│   ├── CTaskBench/
│   │   ├── engine.py                 # Open-loop, closed-loop, and serial drivers
│   │   ├── platform/llm/pdgraph.py   # PDGraph representation
│   │   ├── tasks/                    # Compound LLM applications
│   │   └── utils/                    # Dataset, Docker, DNN, and search helpers
│   ├── Bayes/                        # Bayesian models and profiling data
│   ├── Datasets/                     # Benchmark datasets and task profiles
│   ├── example/                      # Small CTaskBench examples
│   └── evaluation/                   # Paper experiment and plotting scripts
└── vllm/
    ├── vllm/coinference/             # Compound-inference tracking and prediction
    ├── vllm/core/policy.py           # Hermes and baseline scheduling policies
    ├── vllm/engine/arg_utils.py      # Hermes server options
    └── examples/                     # Chat templates and vLLM examples
```

CTaskBench includes Factool, Graph of Thoughts, HuggingGPT, LangChain map-reduce, ReAct, multi-turn conversation, and code-feedback workloads. Some workloads need Docker, external datasets, or API credentials such as `SERPER_API_KEY`; check the selected example before running it.

## Installation

Hermes targets Linux systems with NVIDIA GPUs. The included vLLM fork requires Python 3.8 or newer and pins PyTorch 2.3.0. Python 3.10 is recommended because the CTaskBench dependency set includes newer packages.

Install the modified serving stack and CTaskBench in the same environment:

```bash
git clone https://github.com/NephrenCake/Hermes.git
cd Hermes

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

cd vllm
python -m pip install -e .

cd ../CTaskBench
python -m pip install -e .
```

Building vLLM compiles CUDA extensions. Make sure the installed CUDA toolkit, compiler, driver, and PyTorch build are compatible before installation.

## Running Hermes

### Start the server

From the `vllm` directory, start an OpenAI-compatible server with the Hermes scheduler:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --served-model-name gpt-3.5-turbo \
  --chat-template examples/template_alpaca.jinja \
  --coinference-scheduler \
  --scheduling-policy Hermes \
  --bayes-prediction
```

The server listens on `http://localhost:8000/v1` by default. Replace `/path/to/model` with a local model directory or a supported Hugging Face model identifier.

### Run a benchmark example

In another shell, activate the same environment and run an example from `CTaskBench`:

```bash
cd Hermes/CTaskBench
python example/factool_code.py
```

The examples use the OpenAI-compatible endpoint at `http://localhost:8000/v1`. Edit the selected example if the server address, model name, task rate, or workload size differs.

### Reproduce the paper experiments

Experiment launchers are in `CTaskBench/evaluation/`:

```text
start_sched_e2e_evaluation.py      End-to-end scheduling evaluation
start_sched_gittins_evaluation.py  Gittins-policy evaluation
start_kvc_evaluation.py            KV-cache policy evaluation
start_lora_evaluation.py           LoRA policy evaluation
start_prefetch.py                  Backend prewarming evaluation
start_overhead_evaluation.py       Scheduler overhead evaluation
```

These scripts record the original experimental setup and contain machine-specific model paths, GPU selections, chat-template paths, output directories, and startup delays. Update those values before running a launcher. For example:

```bash
cd CTaskBench/evaluation
python start_sched_e2e_evaluation.py
```

## Citation

If you use Hermes, please cite:

```bibtex
@article{liu2026hermes,
  author  = {Yifei Liu and Zuo Gan and Zhenghao Gan and Weiye Wang and Chen Chen and Yizhou Shan and Xusheng Chen and Zhenhua Han and Yifei Zhu and Shixuan Sun and Minyi Guo},
  title   = {Hermes: Efficient Serving of LLM Applications with Probabilistic Demand Modeling},
  journal = {ACM Transactions on Architecture and Code Optimization},
  volume  = {23},
  number  = {2},
  pages   = {1--25},
  year    = {2026},
  doi     = {10.1145/3803390}
}
```

## License

Hermes is released under the [Apache License 2.0](LICENSE). The serving implementation is based on [vLLM](https://github.com/vllm-project/vllm); upstream copyright and license notices remain in the `vllm/` tree.

## Contact

Questions and feedback are welcome at [nephrencake@sjtu.edu.cn](mailto:nephrencake@sjtu.edu.cn).
