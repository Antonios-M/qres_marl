# QRES-MARL: A Resilience-Based Multi-Agent Reinforcement Learning Framework for Post-Earthquake Recovery of Interdependent Infrastructures

This repository provides means for analysing existing or custom infrastructure testbeds, predicting income losses, relocation costs, traffic delay costs and repair costs and uses MARL for finding near-optimal post-earthquake recovery strategies. The methods used to model and create testbeds as well as predict losses are novel, while the code for RL training is a minimally adapted version of Prateek Bhustali's IMPRL repository (see references). Testbeds are generated using INCORE's [https://incore.ncsa.illinois.edu/doc/pyincore/] data schemas with data from  NBI (National Bridge Inventory), NSI (National Structures Inventory) and OSM. Costs and losses are calculated using various methods from HAZUS and FHWA and other sources which are cited locally; these can be found in _quake_envs_pkg\quake_envs\simulations\_

The work was developed as part of my MSc Thesis in TU Delft and is intended for further use and developement in simillar research applications. Trained agent data on the testbeds used can be provided upon request. View the full pdf manuscript at [https://jmp.sh/k2yp2uBs]


# References

Code used for RL training is located in _imprl-infinite-horizon/_ and is a minimally adapted version of Prateek Bhustali's repository for RL in Inspection and Maintenance Planning:

```bib
@misc{bhustali_impr,
    author= {Prateek  Bhustali},
    title = {Inspection and Maintenance Planning using Reinforcement Learning (IMPRL)},
    howpublished = {GitHub},
    year = {2024},
    url = {https://github.com/omniscientoctopus/imprl}
}
```

Code used for static traffic assignment is located in _quake_envs_pkg\quake_envs\simulations\traffic_assignment.py_ and is a wrapper built on top of Matteo Bettini's repository:

```bib
@misc{Bettini2021Traffic,
  author =       {Matteo Bettini},
  title =        {Static traffic assignment using user equilibrium and system optimum},
  howpublished = {GitHub},
  year =         {2021},
  url =          {https://github.com/MatteoBettini/Traffic-Assignment-Frank-Wolfe-2021}
}
```
