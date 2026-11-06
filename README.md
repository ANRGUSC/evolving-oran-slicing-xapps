# Evolving Inspectable O-RAN Slicing xApps with LLMs

Code and data for the paper *Evolving Inspectable O-RAN Slicing xApps with LLMs*
(Faezeh Dehghan Tarzjani and Bhaskar Krishnamachari, University of Southern California),
arXiv:2609.27337, https://arxiv.org/abs/2609.27337.

We use an LLM to evolve O-RAN slicing controllers as short Python programs. Candidates are
scored in a simulator calibrated to the NSF POWDER 5G testbed, and the selected program runs
unchanged in the testbed's control path. This repository contains the simulators, the evolved
controllers, the prompts and search records, the testbed logs, and scripts that recompute the
numbers reported in the paper.

## Requirements

Python 3.10 or newer. The reproduction scripts need only the standard library.
`matplotlib` is needed only to redraw Fig. 7, and retraining the PPO comparator needs `torch`,
`stable-baselines3` and `gymnasium` (see `requirements.txt`).

## Reproducing the numbers

```
bash reproduce_all.sh
```

This runs the five scripts below (a few minutes in total, most of it the four-slice
held-out evaluation) and prints each recomputed value next to the value in the paper. You can
also run any one of them from inside its folder with `python reproduce.py`.

| Folder | Paper | What it recomputes |
|---|---|---|
| `two_slice` | Sec. V-C, V-E | evolved controller 0.9398 vs the 0.9300 static allocation; the 144 zero-shot programs (best 0.9054, none above 0.9300) and the void rates; the evolution runs; 13 of 20 confident-prompt programs; the release-timing predictions on the 75-step CQI fixture |
| `editability` | Sec. V-G | the one-line calibration edit: SLA misses 2.4% to 79.9% under a 10% efficiency error, 2.2% after the edit, best-effort -6.3 Mbps |
| `hardware` | Sec. V-D, V-F, Table I | the POWDER panel: best-effort 158.2 to 228.6 Mbps, +44.5%; the five PPO seeds, 0.9534 +/- 0.0208 |
| `four_slice` | Sec. VI, Tables II and III, Fig. 6 | the five starting programs and the reference policies; the sealed prompt against the call journals; Table III, selection and held-out, for all 70 selected programs; the Fig. 6 curves |
| `source_repair` | Sec. VI-E, Fig. 7 | the one-constant repair: fitness 0.3226 to 0.7255 on all 16 traces |

## What is where

`two_slice/`
`engine/simulator.py` is the two-slice simulator (Eq. 8 physics and the demand schedule),
`observation_model.py` the CQI observation model and worst-floor fitness, `scorer.py` the
scorer used during evolution, and `cqi_fixture.py` the 75-step CQI fixture of Sec. V-E.
`controllers/` has the evolved controller (`evolved_claude.py`) and the comparison controller
(`comparison_gemini.py`). `data/zero_shot_scores.jsonl` has the fitness and action trace of
every zero-shot call (144 valid programs, 72 per model, and 135 voided calls),
`data/search_history/` the per-generation records of the Claude and Gemini evolution runs, and
`data/cold_draws_confident.jsonl` the 20 initial programs drawn under the confident prompt.
`search/prompt.txt` is the task prompt, `search/prompt_hedged.txt` the hedged variant, and
`search/evolve.py` the evolution loop, kept for reading: it calls the API through a small
client helper and a seed program that are not part of this release.

`editability/`
`patches/E1_provisioning_margin.diff` is the one-line edit of Sec. V-G. `reproduce.py` applies it
to the evolved two-slice controller and re-evaluates both versions on the 96 development
episodes under the nominal delivery model and under a 10% lower good-channel efficiency.

`hardware/`
`logs_312891_rep1/` has the per-tick KPM logs of the five controllers under the good channel
(`rep1_A_good_demand_*`) and the deep null (`rep1_B_channel_null_*`), about 1 Hz for 120 s each.
`ppo_baseline/` has the deployed PPO checkpoint (`ppo_policy.zip`, the seed-42 agent), the
recorded scores of the five training seeds (`ppo_experiment_records.json`), the trainer and
its environment.

`four_slice/`
`engine/simulator.py` is the four-slice simulator (Eq. 11 efficiency curve, staggered fades,
one-tick observation lag) and `engine/scorer.py` the fitness (worst-env composite).
`controllers/` holds the five starting programs: the fixed-threshold baseline, the hand-written
expert rule (running it re-runs the nine-setting grid of Sec. VI-B), the greedy seed, the
prompting seed and the champion seed. `analysis/anchors.py` recomputes the reference policies.
`search/prompt.txt` is the sealed task prompt (its SYSTEM and USER blocks hash to the values
journaled on every API call) and `search/evolve.py` the search loop (Algorithm 1), kept for
reading: the API client and cost ledger it imports are not part of this release.
`data/runs/` has the complete archives of the ten runs of each primary arm of Table III
(64 proposals each; the `.jsonl` file of a run holds every candidate with its source and score,
the `.prompts.jsonl` file the hashes of every API call). `data/held_out/selected_sources/` has
the program returned by each of the 70 runs, `held_out_seeds.json` the 200 sealed traces and
`held_out_scores.json` the recorded selection and held-out score of each run.
`data/search_histories.json` is the best-so-far fitness of every run in Fig. 6.

`source_repair/`
`inputs/controller.py` is the saved controller from the prompting-seeded arm. `repair.py`
applies the 11.8 to 2.19 edit, replays both versions with the four-slice simulator and scorer
(unchanged) and writes `results/`; `plot_fig7.py` draws Fig. 7.

The LLM runs used `claude-haiku-4-5-20251001` at temperature 1.0 (two-slice comparison:
`gemini-2.5-flash-lite`).

## Citation

```bibtex
@misc{dehghantarzjani2026evolving,
  title         = {Evolving Inspectable O-RAN Slicing xApps with LLMs},
  author        = {Dehghan Tarzjani, Faezeh and Krishnamachari, Bhaskar},
  year          = {2026},
  eprint        = {2609.27337},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SY},
  url           = {https://arxiv.org/abs/2609.27337}
}
```

Questions: dehghant@usc.edu
