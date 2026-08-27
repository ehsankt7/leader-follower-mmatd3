# Running the experiments

This page provides the command reference for the released implementation. Methodological details and experimental analysis are described in the associated paper.

## Python setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
pytest -q
```

## ROS/Gazebo workspace

After installing and sourcing a compatible ROS 1/Gazebo stack:

```bash
cd ros/catkin_ws
catkin_make
source devel/setup.bash
cd ../../..
```

## Simple-environment experiments

```bash
python scripts/train.py --config configs/simple_mmatd3.json --output outputs/simple_mmatd3
python scripts/train.py --config configs/simple_matd3.json  --output outputs/simple_matd3
python scripts/train.py --config configs/simple_maddpg.json --output outputs/simple_maddpg
```

## Complex-environment experiment

```bash
python scripts/train.py --config configs/complex_mmatd3.json --output outputs/complex_mmatd3
```

A saved simple-environment checkpoint can optionally be supplied:

```bash
python scripts/train.py \
  --config configs/complex_mmatd3.json \
  --initialize-from outputs/simple_mmatd3/checkpoints/mmatd3_simple_final.pt \
  --output outputs/complex_mmatd3
```

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/complex_mmatd3.json \
  --checkpoint outputs/complex_mmatd3/checkpoints/mmatd3_complex_final.pt \
  --episodes 100 \
  --output outputs/complex_mmatd3/evaluation.csv
```

## Plotting

```bash
python scripts/plot_metrics.py --help
```

The JSON files in `configs/` define the released experiment settings.
