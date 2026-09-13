# Diagnosability-Bounded Admissibility Verification for LLM Agents in Industrial Cyber-Physical Systems

Code, result files and raw language-model outputs for the manuscript of the same title by Chenxuan Zhou and Christy Jie Liang (School of Computer Science, University of Technology Sydney), submitted to *IEEE Transactions on Industrial Cyber-Physical Systems*.

The paper places a verification layer between a language-model agent that proposes fault hypotheses and a plant operator. The layer uses the installed sensors and a structural model of the plant to accept, reject or escalate each proposal. It is evaluated on the Tennessee Eastman process and on the voraus-AD robot dataset.

## Contents

| path | what it contains |
|---|---|
| `src/tep/` | Tennessee Eastman structural model, physical constants, simulator build script, data generation |
| `src/robot/` | Structural model of the six-axis robot, voraus-AD loader, robot experiments |
| `src/structural/` | Structural diagnosability analysis shared by both systems |
| `src/verifier/` | Residual generators, fault signature matrix, conformal calibration, verification operator, separation procedure |
| `src/agents/` | Language-model client, baselines B0 to B7, proposition catalogue, learned detector |
| `src/eval/` | Experiments E3 to E10, stratified analysis, figures |
| `results/` | JSON and CSV files behind every number in the paper |
| `figs/` | Figures, as PNG and as vector PDF |
| `logs/llm/` | Every language-model call: prompt messages, raw response, model identifier, token usage, latency and timestamp, one JSON object per line, gzip-compressed |
| `data/tep/` | Metadata of the generated simulation runs (`spec.json`, `*_meta.json`) |
| `run` | Master script for Code Ocean; also runs locally |
| `compare_results.py` | Compares regenerated result files with the stored ones |

## Running on Code Ocean

Select `run` in the repository root with **Set as File to Run**. Use a Python 3.12 environment with the pip packages `numpy==2.5.2`, `matplotlib==3.11.1` and `faultdiagnosistoolbox==0.12.5`.

- Without attached data, the script runs Gate A, Gate B, E1, the residual-bank separation procedure, E7 and the figures, in about one minute.
- If the Tennessee Eastman arrays are attached as data, either `tep_simulation_runs.zip` from IEEE DataPort or its extracted `data/tep/` folder, the script also runs E2, E5, E6, E9, E10, the shutdown-policy comparison and the stratified results table. This takes about five more minutes on a desktop CPU.
- Outputs are written to `/results`: the regenerated `results/` and `figs/`, `run_log.txt`, and `comparison.txt`, which compares each regenerated JSON file with the stored copy in this repository. E10 latencies depend on the hardware and are not compared.
- The simulator build, data generation, robot experiments and language-model experiments are not run on Code Ocean. Their stored outputs are in `results/` and `logs/llm/`.

The same script runs locally, for example `DATA_DIR=data/tep RESULTS_DIR=out bash run`.

## Not included

- **API key.** Hosted models are called through the Moonshot OpenAI-compatible endpoint. Put your own key in `key.txt` in the repository root; the file is ignored by git and the key is never logged.
- **Tennessee Eastman source code and reference data.** Place `teprob.f`, `temain_mod.f`, `d00.dat`, `d00_te.dat` and `d01_te.dat` in `data/raw/tep_src/`. They are the Fortran codes and data files distributed by the Braatz group, available for example from [camaramm/tennessee-eastman-profBraatz](https://github.com/camaramm/tennessee-eastman-profBraatz).
- **voraus-AD dataset.** Download the 100 Hz parquet file from [vorausrobotik/voraus-ad-dataset](https://github.com/vorausrobotik/voraus-ad-dataset) and save it as `data/raw/voraus/voraus-ad-100hz.parquet`. The dataset is licensed CC BY-NC-SA 4.0.
- **Generated arrays and the compiled simulator.** The simulation arrays (`data/tep/*.npy`, about 325 MB), the extracted robot recordings (`data/voraus/`, about 315 MB) and the compiled simulator are produced by the scripts below.

## Environment

- Python 3.12 with the packages in `requirements.txt`
- gfortran 13.2 (MinGW-w64 on Windows) for building the simulator through f2py, meson and ninja
- Ollama, for the open-weight models `qwen3.5:9b`, `qwen2.5:7b` and `llama3.1:8b`

```
pip install -r requirements.txt
```

## Reproduction

Run the commands from the repository root, in this order.

```
# structural analysis, no data needed
python src/tep/run_gateB.py
python src/robot/run_gateA.py
python src/structural/run_E1.py
python src/structural/run_E1_augment.py

# simulator
python src/tep/build_sim.py
python src/tep/validate_sim.py

# data
python src/tep/gen_data.py --what core
python src/tep/gen_data.py --what sweeps
python src/tep/gen_sweep_extra.py
python src/tep/gen_sweep_full.py
python src/tep/gen_stiction.py
python src/robot/voraus_data.py

# verification layer and experiments that need no language model
python src/verifier/run_E2.py
python src/verifier/annihilators.py
python src/eval/run_E5.py
python src/eval/run_E6.py
python src/eval/run_E9.py
python src/eval/run_E10.py
python src/eval/run_shutdown_policy.py
python src/robot/diagnose_isolation.py
python src/robot/run_robot.py

# language-model experiments (hosted models are billed)
python src/eval/run_E3E4.py --models kimi-k3,kimi-k2.6 --baselines B0,B5L,B6,B7 --seeds 3 --per-location 6 --nf 12 --tag final --resume
python src/eval/run_E8.py --models kimi-k3 --seeds 3 --per-location 20
python src/eval/run_E3E4.py --models kimi-k3 --baselines B4 --seeds 2 --per-location 6 --nf 12 --tag b4_114

# analysis and figures
python src/eval/run_strata.py --per-location 6 --nf 12 --grids E4_final,E4_final_local --tag final
python src/eval/run_E7.py E4_final
python src/eval/make_figs.py
```

The open-weight runs use the same `run_E3E4.py` script with the model aliases defined in `src/agents/llm.py`. The language-model experiments were run on 11 and 12 September 2026; hosted model versions change over time, so re-running them may not reproduce the logged outputs exactly, which is why every raw output is provided in `logs/llm/`.
