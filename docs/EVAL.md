# Running Policy Evaluation in OpenWorld

Policy evaluation requires the following:

* Test suite with different test cases
* A trained policy (dp or openpi)
* (Optional) A reward function for success annotations

## Creating a test suite

Each test case requires the following input:

- Initial observation from the left, right, and wrist-view camera
- Initial robot state + gripper position
- (Optional) Task instruction

We provide an example test suite for the droid setup at [data/evaluation_suites/irom_test_carrot](data/evaluation_suites/irom_test_carrot). Feel free to create your own!

## Setting up Reward Function

```bash
# Robometer
git clone https://github.com/robometer/robometer.git external/robometer
```

## Running Evaluation

Finally, to generate evaluation videos with reward annotations, run:

```bash
# Example: openpi
uv sync --extra openpi --extra robometer
uv run python scripts/run_evaluation.py \
  --config configs/evaluation/example_eval_openpi.yaml
```

Output will contain the following:
