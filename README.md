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

## Repository Structure and Contents

- two_slice/ contains the two-slice simulator and evaluation framework. engine/simulator.py implements the physical model in Eq. (8) and the demand schedule; observation_model.py defines the CQI observation model and worst-floor fitness; scorer.py evaluates candidates during evolution; and cqi_fixture.py provides the 75-step CQI fixture used in Sec. V-E. The controllers/ directory contains the evolved controller (evolved_claude.py) and the comparison controller (comparison_gemini.py).
  data/zero_shot_scores.jsonl records the fitness and action traces for the zero-shot experiments, covering 144 valid programs (72 per model) and 135 voided calls. data/search_history/ contains generation-level records from the Claude and Gemini evolution runs, while data/cold_draws_confident.jsonl contains the 20 initial programs generated using the confident prompt. The task prompt and its hedged variant are provided in search/prompt.txt and search/prompt_hedged.txt, respectively. search/evolve.py documents the evolution loop but requires an API client helper and a seed program that are not included in this release.
- editability/ contains the one-line modification evaluated in Sec. V-G, provided as patches/E1_provisioning_margin.diff. The reproduce.py script applies this patch to the evolved two-slice controller and evaluates both the original and modified versions on the 96 development episodes. Evaluation covers the nominal delivery model and a perturbed model with a 10% reduction in good-channel efficiency.
- hardware/ contains the hardware experiment logs and PPO baseline artifacts. logs_312891_rep1/ provides per-tick KPM measurements for all five controllers under the good-channel condition (rep1_A_good_demand_*) and the deep-null condition (rep1_B_channel_null_*). Each measurement sequence spans approximately 120 seconds at a sampling rate of 1 Hz. ppo_baseline/ contains the deployed seed-42 PPO checkpoint (ppo_policy.zip), recorded scores for the five training seeds (ppo_experiment_records.json), and the training script and environment.
- four_slice/ contains the four-slice simulator, controllers, and evaluation records. engine/simulator.py implements the efficiency curve in Eq. (11), staggered fades, and a one-tick observation lag. engine/scorer.py computes the composite fitness based on the worst-performing environment. The controllers/ directory contains five starting programs: the fixed-threshold baseline, the hand-written expert rule, the greedy seed, the prompting seed, and the champion seed. Executing the expert-rule script repeats the nine-setting grid evaluation described in Sec. VI-B. analysis/anchors.py recomputes the reference-policy results.
  search/prompt.txt contains the sealed task prompt. The hashes of its SYSTEM and USER blocks match those recorded for each API call. search/evolve.py implements the search loop in Algorithm 1 and is included for inspection; its API client and cost-ledger dependencies are not included in this release.
  data/runs/ contains complete archives for the ten runs of each primary arm in Table III, with 64 proposals per run. Each run’s .jsonl file records candidate source code and scores, while its .prompts.jsonl file records the prompt hashes for every API call. data/held_out/selected_sources/ contains the program selected by each of the 70 runs. Within data/held_out/, held_out_seeds.json defines the 200 sealed evaluation traces, and held_out_scores.json records each run’s selection and held-out scores. data/search_histories.json provides the best-so-far fitness trajectories shown in Fig. 6.
- source_repair/ contains the source-level repair experiment. inputs/controller.py stores the controller selected from the prompting-seeded arm. repair.py changes the constant from 11.8 to 2.19, evaluates both versions using the unchanged four-slice simulator and scorer, and saves the outputs to results/. plot_fig7.py generates Fig. 7.
The LLM experiments used claude-haiku-4-5-20251001 at a temperature of 1.0. The two-slice comparison used gemini-2.5-flash-lite.

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
