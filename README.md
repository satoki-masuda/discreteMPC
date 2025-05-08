# Predictive Traffic Control for Evacuation Contraflow using ZDD Sampling

- This is a Python implementation of Masuda and Hato (2025) "Predictive Traffic Control for Evacuation Contraflow using ZDD Sampling".

## Requiremnts
- Python >=3.10
- Poetry 2.1.1

## Installation

```bash
poetry install
```

## Main files
- `discrete_mpc/model/discrete_mpc.py`: run discrete-variable MPC using cross-entropy method with ZDD sampling
- `discrete_mpc/model/sampling_compare.py`: compare sampling methods

## Function files
- `discrete_mpc/model/mfd_dynamics.py`: simulate traffic dynamics using MFD model
- `discrete_mpc/model/reconf_horizon.py`: calculate reconfiguration constraints using ZDD

## Setting files
- `discrete_mpc/model/parameters_ndp.py`: define parameters for simulation

## Analysis files
- `discrete_mpc/model/calculation.ipynb`: analyze the calcualted results
