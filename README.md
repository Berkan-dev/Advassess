<p align="center">
  <img src="assets/kraken.png" alt="Advassess kraken badge" width="300">
</p>

<h1 align="center">Advassess</h1>

Advassess tests how easily AI models can be fooled or manipulated, and reports the results. You work with it through an AI
agent: the agent writes the assessment config, shows you the plan, runs the attacks once you approve, and reads back the report.

Use it only on models and endpoints you own or are explicitly authorized to test.

## What it does

It runs adversarial attacks against:

- **Image and numeric classifiers**: PyTorch (TorchScript, or weights plus the model class), Keras / TensorFlow (`.keras`, `.h5`)
  and scikit-learn models.
- **Attacks that need more than a model and data.** These are in `--list-attacks` under "how it runs":

- *One shared perturbation* (`universal`, `targeted_universal`, `patch`, `adversarialpatch*`): one perturbation or patch is learned on
  part of your data and scored on the held-out rest, because a universal attack that is judged on its own training inputs
  proves nothing. They need at least 20 samples; patches need image input.
- *Setup derived by the tool* (`lowprofool`, `featureadv*`, `graphite*`, `laser`): the extra inputs these attacks require (feature
  importances, guide images, target class, patch and network sizes) are worked out from your data and model. If the model type
  cannot support the attack, the run says so instead of guessing.
- *Adapter* attacks (object detection, tracking, video, speech, malware, Gaussian-process and decision-tree-leaf models): the model
  and its scoring are not something a plain config can describe. Write a small adapter class (`python run.py --adapter-help`),
  and set `target: {source: adapter, adapter: "my_adapter.py:MyAdapter"}`. The adapter file runs with your permissions, like
  `target.model_class`. The adapter's `score()` must return `clean_score`, `adv_score` and `success_rate`.

Verified during development: `universal`, `targeted_universal`, `lowprofool`, `dtattack` and the adapter plumbing, on scikit-learn
models. Not run during development (they need PyTorch, TensorFlow or real detection, video or speech models): the patch attacks,
`featureadv*`, `graphite*`, `laser` and every adapter-driven attack. Treat the first run of these as untested.

**Text classifiers**: scikit-learn pipelines that take raw text.
- **Language models**: local or hosted, through Garak (fixed jailbreak and injection probes) and PyRIT (an attacker LLM that
  converses with the target and adapts each turn).

Every run ends by writing `report.html` (open it in a browser). It is not optional: if it cannot be written the run exits with an error, and if one chart fails the page is still written and says which. Each run produces a results table, charts, example attack inputs and a self-contained `report.html`. Results list the model's
accuracy before and after each attack, how many inputs were fooled, warnings (for example an attack that changed nothing, which
would otherwise look like robustness), attacks that were skipped, and the limits of the assessment.

Scikit-learn pickles and PyTorch pickles can run code when loaded, so they are refused unless you set
`ADVASSESS_ALLOW_UNSAFE_MODEL_LOADING=1` for a trusted file. Prefer TorchScript or weights.

## Attacks

If no attacks are named, a built-in suite runs, sized to the data and skipping attacks that do not apply to the model.

**Classifiers** (28 attacks from the Adversarial Robustness Toolbox):

| Type | Attacks |
|---|---|
| White-box (use the model's gradients) | FGSM, BIM, PGD, MIM, APGD, ACG, RACG, VAT, Shadow, Composite, Carlini & Wagner (L2, L-inf, L0), DeepFool, NewtonFool, ElasticNet (EAD), JSMA, Wasserstein |
| Black-box (only query the model) | Boundary, HopSkipJump, SignOPT, Square, ZOO, SimBA, Pixel, Threshold, Spatial transformation |
| Mixed | AutoAttack |

Untargeted by default; most attacks also support a targeted mode. Shadow and Composite were not verified during development.

**Text classifiers**: `char_perturb` (misspells the words the model relies on most) and `word_insertion` (appends words typical of
other classes). Both only query the model.

**Language models**:

- Garak: its full probe library, covering jailbreaks, prompt injection, encoding tricks, latent injection, data leakage, toxicity and
  more. Probes are chosen by module, by name or by tag.
- PyRIT strategies (set `strategy:` on an `engine: pyrit` attack):

  | Strategy | What it does | Needs attacker model |
  |---|---|---|
  | `prompt_sending` | sends each objective as one prompt | no |
  | `red_teaming` | attacker LLM converses with the target, adapting every turn | yes |
  | `crescendo` | gradual escalation over several turns | yes |
  | `many_shot` | Many-Shot Jailbreak: a long faux dialogue of an "AI" complying is placed before the objective (`example_count`, at least 50, default 100) | no |
  | `skeleton_key` | Skeleton Key: first persuades the model to "update" its guidelines and answer with a warning instead of refusing, then sends the objective | no |
  | `pair` | PAIR: parallel attacker streams (`tree_width`, at least 3) each refine their prompt from the last reply; `max_turns` is the depth | yes |
  | `tap` | Tree of Attacks with Pruning: PAIR plus branching and pruning of weak attempts | yes |

  Success is judged by a judge model by default; a plain substring match exists only as a sanity check (not available for `pair`
  and `tap`, which rank attempts by a 0-1 judge score with PyRIT's built-in "task achieved" scale at threshold 0.7). Each objective is
  attempted several times (`repeats`) so the result has a sample size. `crescendo`, `many_shot`, `skeleton_key`, `pair` and `tap` were
  not run against a real model during development: check the first transcripts before trusting the numbers.

**Language-model scans always run at full strength**: every prompt of each Garak probe, several replies each, and multi-turn attacks
with room to adapt. Settings can be raised but not lowered below fixed minimums, and there are no quick or standard profiles.

**How results are judged** (the same for every model, probe and language):

- **Evidence, not a bare percentage.** Each result shows a 95% confidence interval over its independent prompts or attempts. A clean
  result with too few of them is labelled **inconclusive** (zero failures in a handful of prompts is compatible with a high real
  failure rate), and `plan_assessment` warns about it up front.
- **Reviewer model.** Detectors fire on surface patterns, so they also flag replies that refuse, correct or quote the trigger. Set a
  `review` model (any OpenAI-compatible endpoint, different from the model under test) and every flagged reply is read by it in
  context. Likely false positives are labelled and excluded from the verdict, never deleted from the raw counts. Without a reviewer,
  flags are reported as **unverified**, not guessed at.
- **Role checks.** Plans and reports warn when the attacker, judge or reviewer is the model under test, and remind the agent that the
  attacker must be stronger than the target.

## Quick install

Requires Python 3.10 or newer.

```bash
python -m pip install --user pipx
python -m pipx ensurepath          # then open a new terminal
pipx install "git+https://github.com/Berkan-dev/advassess.git"
advassess doctor
```

Add the engines you need to the same environment:

```bash
pipx inject advassess torch adversarial-robustness-toolbox matplotlib        # PyTorch and scikit-learn numeric models
pipx inject advassess tensorflow adversarial-robustness-toolbox matplotlib   # Keras / TensorFlow models
pipx inject advassess garak                                                  # language-model probes
```

PyRIT must live in its own Python environment (`python -m venv pyrit-env`, then `pyrit-env/bin/pip install pyrit`), because it and
the Adversarial Robustness Toolbox both install a package named `art` and overwrite each other. The agent points the config's
`pyrit.python` at that environment's interpreter.

## Using it with agents

Advassess runs as an MCP server. One command configures your agents:

```bash
advassess setup --workspace /path/to/your/workspace
```

It supports Claude Code, Codex, Cursor, Gemini CLI and Claude Desktop. It shows exactly what it will change, asks for approval, backs
up each existing config file and edits only its own `advassess` entry. Restart the agent afterwards. Keep your models, data and
results inside the workspace folder. To remove it: `advassess uninstall`.

Then ask your agent in plain language, for example: *"Assess the Keras model in `models/classifier.keras` against the test data in
`data/test.npz`."* The agent will:

1. write the assessment config for you (`write_config`) and describe the plan (`plan_assessment`), including any request counts
   and costs for hosted models;
2. wait for your approval, then start the run in the background (`start_assessment`) and follow it (`get_job`);
3. read back the results (`get_report`) and explain them.

| Tool | Purpose |
|---|---|
| `list_attacks`, `describe_attack`, `list_probes` | What can be run, and each attack's settings |
| `write_config`, `read_config` | Create or read the assessment config |
| `plan_assessment` | Show what would run, without running anything |
| `start_assessment`, `get_job`, `cancel_job` | Run in the background, check progress, stop |
| `list_runs`, `get_report` | Find finished runs and read their results |

API keys are never put in a config: name an environment variable instead, and set it where the agent is started.
