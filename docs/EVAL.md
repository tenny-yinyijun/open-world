# Running Policy Evaluation in OpenWorld

## Creating a test suite

We provide an example DROID test suite at [data/evaluation_suites/irom_test_carrot](data/evaluation_suites/irom_test_carrot). Each test case requires the following input:

- Initial observation from the left, right, and wrist-view camera
- Initial robot state
- (Optional) Task instruction

## Running evaluation

Evaluation setup in openworld is specified by a single config file and can be ran with:

```bash
uv run python scripts/run_evaluation.py \
  --config configs/evaluation/example_eval_openpi.yaml
```

 Important fields to edit in the config:
- `num_inference_steps`: shortcut model steps for faster inference (1, 2, 4, 16, 50). Inference speed with quality tradeoff.
- `policy`: currently supports dppo and openpi interfaces
- `duration`: duration of the generated video (in seconds)