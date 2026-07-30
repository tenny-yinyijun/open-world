# Scene Generation — building eval suites from a base + edits

`openworld.scenegen` turns a **base scene** plus a **list of edits** into an
**Initialization suite** — the format the policy-eval pipeline consumes
(`scripts/run_evaluation.py`, see [EVAL.md](EVAL.md)).

```
base (unedited views + robot start state)  +  edits (prompt + instruction)
   ▼
[1] guardrail   rewrite instruction -> editor-ready prompt, then specialize per camera
[2] mode        generate the edited views (nanobanana / multiview / copy)
[3] suite       write initialization.yaml + views, then verify against the eval loader
   ▼
<suite>/init_*/{<view>.png ...} + initialization.yaml
```

Point an eval config's `dataset_path` at the suite and run it — **scene
generation produces the benchmark that [policy evaluation](EVAL.md) runs on.**

---

## Layout

| Piece | Path |
|-------|------|
| modes (the pluggable generative half) | `openworld/scenegen/modes/` |
| prompt layer (guardrail + per-view + chaining) | `openworld/scenegen/guardrail.py` |
| view sets / bases | `openworld/scenegen/views.py` |
| suite writer + eval-format contract | `openworld/scenegen/suite.py` |
| orchestration | `openworld/scenegen/builder.py` |
| **suite CLI** | `scripts/scenegen/build_suite.py` |
| **add-object CLI** (GPU) | `scripts/generate_test_case.py` |
| bases (`tri`, `irom`) | `assets/<base>/` |
| suite specs | `configs/scenegen/suites/` |
| tests | `openworld/scenegen/tests/test_scenegen.py` |

---

## Modes

nanobanana is **one mode among several**. A mode only makes pictures; suite
layout, the `initialization.yaml`, and verification are shared by all of them.

```bash
python scripts/scenegen/build_suite.py --list-modes
```

| Mode | What it does | Needs | Anchor orders |
|------|--------------|-------|---------------|
| `nanobanana` | chained per-view edit with Gemini 2.5 Flash Image. Edits views that already exist — background, lighting, material, recoloring. | `GOOGLE_API_KEY` | both |
| `multiview` | nanobanana edits the wrist view, FLUX.2-klein **synthesizes** the sides. Use to introduce a genuinely new object. | GPU + diffusers fork + checkpoint | `wrist_first` only |
| `copy` | passes the base through unedited — control case / offline smoke test. | nothing | both |

**Adding a mode.** Subclass `SceneGenMode`, implement `generate_case`, and
decorate with `@register_mode`. The CLI, the spec format, and the suite writer
pick it up with no further changes:

```python
from openworld.scenegen.modes.base import CaseResult, SceneGenMode, register_mode

@register_mode
class MyMode(SceneGenMode):
    name = "mymode"
    description = "..."
    def generate_case(self, *, base, edit, case_dir, edit_order):
        ...  # write <case_dir>/<view>.png for each view in base.view_set.views
        return CaseResult(in_place=True, metadata={"edit_mode": self.name})
```

---

## View sets: 3-view and 2-view

A **view set** is the ordered cameras a world-model checkpoint was trained on.
The canonical order is always *sides first (right, then left), wrist last* —
exactly the `view_order` the eval configs declare.

| Views | `view_order` | Model | Eval config |
|-------|--------------|-------|-------------|
| 3 | `[exterior_right, exterior_left, wrist]` | 3-view DROID | `configs/evaluation/0617_ar_pi05.yaml` |
| **2** | `[exterior_right, wrist]` | `wm_student_2view` | `configs/evaluation/teleop_ar_pi05.yaml` |

Select the subset with `views:` in a spec or `--views` on the CLI. A 3-view base
can build a 2-view suite — it just uses the cameras you asked for:

```bash
GOOGLE_API_KEY=... python scripts/scenegen/build_suite.py \
    --spec configs/scenegen/suites/example_2view.yaml \
    --views exterior_right wrist
```

---

## Edit order: which view anchors

Editing views independently makes the editor **re-invent the scene differently in
each camera** — a new object lands in a different spot, or changes color. So views
are edited as a *chain*: the **anchor** is edited first from the prompt alone, and
every later view is edited with the already-edited views passed in as reference
images plus an explicit "reproduce that same scene from this viewpoint" clause.

Both orders are supported, and which one you want depends on the edit:

| `edit_order` | Chain (2-view) | Use for |
|--------------|----------------|---------|
| `wrist_first` (default) | `wrist` → `exterior_right` | **object** edits — add / remove / replace / recolor. The overhead view pins placement on the table most precisely. |
| `side_first` | `exterior_right` → `wrist` | **background / lighting / room** edits. The side view is where the room actually is, so anchor there. |

Set it suite-wide (`edit_order:`) or per-edit (`edit_order:` inside an edit), or
override on the CLI with `--edit-order`. `multiview` is wrist-anchored by
construction and rejects `side_first` rather than silently ignoring it.

### Prompts are per-camera

The same edit needs different phrasing per camera, or you get an object's top
surface pasted into a side view. `guardrail.py` owns this:

- `SYSTEM_PROMPT` / `build_edit_prompt` — rewrite a plain instruction into an
  editor-ready prompt. Deliberately **viewpoint-neutral**; describes placement in
  terms true from any viewpoint ("to the left of the bowl"), never "top-down".
- `VIEW_GEOMETRY` — the per-camera clause. The wrist entry says *"top-down
  (bird's-eye) … show its top surface as seen from directly above, not its side
  profile"*; the exterior entries say *"viewing the table from the side at an
  angle … show its front and side faces … NOT as a flat top-down cutout"*, and name
  which side the camera is on.
- `chain_prompt` — the "match the reference image(s)" clause for follower views.
- `PRESERVE_CLAUSE` — the invariants no edit may break (camera, framing, gripper).

Two guardrail backends: `gemini` (default, `gemini-2.5-flash` rewrite, falls back
on error) and `template` (deterministic, offline).

---

## Building a suite

```bash
# 2-view, both anchor orders demonstrated:
GOOGLE_API_KEY=... python scripts/scenegen/build_suite.py \
    --spec configs/scenegen/suites/example_2view.yaml

# offline check of plumbing + eval format (no API calls):
python scripts/scenegen/build_suite.py \
    --spec configs/scenegen/suites/example_2view.yaml --mode copy
```

**Spec keys** (`configs/scenegen/suites/example_2view.yaml` is fully commented):

| Key | Meaning |
|-----|---------|
| `base` *(required)* | `tri` / `irom` under `assets/`, or a path to a dir of views |
| `edits` *(required)* | list of cases; each needs `prompt` + `instruction` |
| `name` / `out_dir` | output location (`data/benchmark/<name>`) |
| `views` | view subset, e.g. `[exterior_right, wrist]` |
| `mode` / `mode_params` | scene-generation mode (default `nanobanana`) |
| `edit_order` | `wrist_first` (default) or `side_first` |
| `keep` | shared "keep everything else" clause, appended to every prompt |
| `include_base_case` | prepend an unedited control case (zero generative drift) |
| `start_index` | first case index (append to an existing suite) |

Per-edit keys: `prompt`, `instruction` (the *policy* command — independent of what
the image shows), `label`, `keep` (override), `views` (subset to edit), and
`edit_order` (override the suite's anchor).

Library entry point:

```python
from openworld.scenegen import build_suite, build_suite_from_spec
build_suite_from_spec("configs/scenegen/suites/example_2view.yaml")
```

### Add a new object (GPU)

```bash
GOOGLE_API_KEY=... python scripts/generate_test_case.py \
    --instruction "put the carrot in the bowl" \
    --base tri --views exterior_right wrist --name carrot_2view
```

Prereqs: the diffusers fork (`external/diffusers`, FLUX.2-klein lives only
there), `checkpoints/multiview_droid_v0` (`bash external/download_models.sh`), and
a GPU (~8 GB). `scripts/scenegen/make_suite_add_object.sh` batches this over an
object list; `scripts/scenegen/remove_object.py` chain-erases an object from each
view to make an empty-table base.

---

## Output, and why it loads in eval

```
<suite>/
├── scenegen_manifest.json      # provenance: mode, views, edit_order, per-case prompts
└── init_*/                     # <view>.png (320×192) + initialization.yaml
```

Each `initialization.yaml`:

```yaml
initial_state:                  # cloned from the base's template.yaml
  robot: {...}
initial_observation:            # explicit — see below
  views:
    exterior_right: exterior_right.png
    wrist: wrist.png
instruction: put the mug in the white container
metadata:
  edit_mode: nanobanana
  edit_order: wrist_first
  anchor_view: wrist
  edit_chain: [wrist, exterior_right]
  view_prompts: {...}           # the exact prompt sent per camera
```

**`initial_observation` is always written explicitly, not inferred.** The loader
*can* infer views from a case directory, but
`InitializationDataset._infer_observation_from_case_dir` only does so when **all
three** DROID views are present — it returns `None` for a 2-view case, which would
leave the observation unset and break the rollout. Writing the block explicitly
makes 2-view suites first-class and pins view *order* instead of relying on
directory inference. The names and order match the `view_order` in the eval
configs, which is how `ARWanWorldModel._bootstrap_history` height-stacks cameras.

**Every build is verified.** `suite.verify_suite` loads the finished suite back
through the real `InitializationDataset` and checks each case has a usable
`initial_state` and an `initial_observation` whose view files all exist. A suite
`run_evaluation.py` could not read fails at build time instead of at eval time.
Skip with `--no-verify`.

---

## Tests

```bash
uv run pytest openworld/scenegen/tests/test_scenegen.py
python3 openworld/scenegen/tests/test_scenegen.py   # standalone, no pytest needed
```

Offline (the `copy` mode needs no key), so the suite-building and
eval-compatibility checks are real end-to-end runs.

**Next:** run a policy in a world model over the suite → [EVAL.md](EVAL.md).
