<p align="center">
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  </a>&nbsp;&nbsp;
  <a href="https://networkx.org/">
    <img src="https://img.shields.io/badge/NetworkX-Python-blue?style=for-the-badge&logo=python&logoColor=white" alt="NetworkX" />
  </a>&nbsp;&nbsp;
  <a href="https://gymnasium.farama.org/">
    <img src="https://img.shields.io/badge/Gymnasium-40c4ff?style=for-the-badge&logo=python&logoColor=white" alt="Gymnasium" />
  </a>&nbsp;&nbsp;
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  </a>&nbsp;&nbsp;
  <a href="https://geopandas.org/">
    <img src="https://img.shields.io/badge/GeoPandas-Python-blue?style=for-the-badge&logo=python&logoColor=white" alt="GeoPandas" />
  </a>&nbsp;&nbsp;
  <a href="https://github.com/gboeing/osmnx">
    <img src="https://img.shields.io/badge/osmnx-Python-blue?style=for-the-badge&logo=python&logoColor=white" alt="osmnx" />
  </a>
</p>

# QRES-MARL: A Resilience-Based Multi-Agent Reinforcement Learning Framework for Post-Earthquake Recovery of Interdependent Infrastructures

This repository provides means for analysing existing or custom infrastructure testbeds, predicting income losses, relocation costs, traffic delay costs and repair costs and uses MARL for finding near-optimal post-earthquake recovery strategies. The methods used to model and create testbeds as well as predict losses are novel, while the code for RL training is a minimally adapted version of Prateek Bhustali's IMPRL repository (see references). Testbeds are generated using INCORE's [https://incore.ncsa.illinois.edu/doc/pyincore/] data schemas with data from  NBI (National Bridge Inventory), NSI (National Structures Inventory) and OSM. Costs and losses are calculated using various methods from HAZUS and FHWA and other sources which are cited locally; these can be found in _quake_envs_pkg\quake_envs\simulations\_

The work was developed as part of my MSc Thesis in TU Delft and is intended for further use and developement in simillar research applications. Trained agent data on the testbeds used can be provided upon request. View the full pdf manuscript at [https://jmp.sh/k2yp2uBs]

---

# How to use
1. **Clone the repository**

   ```bash
   git clone https://github.com/Antonios-M/qres_mar.git
   cd qres_mar
   ```

2. **Set up the Conda environment**  
   Make sure you have Conda installed, then create the environment using the `conda_env.yml` located in the **parent folder**:

   ```bash
   conda env create -f ../conda_env.yml
   conda activate qres_marl
   ```


3. **Run inference or training**
   To run inference or training you need saved testbed GeoJSON files for buildings, roads and traffic data. For storing these files see _quake_envs_pkg\quake_envs\simulations\utils.py_.
   Building and Road GeoJSON files should follow the INCORE data schemas [https://incore.ncsa.illinois.edu/doc/pyincore/], traffic csv network and demand files should follow TNTP data
   schemas from [https://github.com/bstabler/TransportationNetworks].
   
   Navigate to the `imprl-infinite-horizon/examples` directory and run the appropriate script:

   - Run **inference** (requires saved trained agents .pth files at wandb/):
     ```bash
     python imprl-infinite-horizon/examples/inference.py
     ```

   - Test basic **training**:
     ```bash
     python imprl-infinite-horizon/examples/train.py
     ```

   - Run **training with logging** using Weights & Biases:
     ```bash
     python imprl-infinite-horizon/examples/train_and_log_parallel.py
     ```

---

## Results
![res_final](https://github.com/user-attachments/assets/45879e2e-d003-478f-94d2-47975bb23fc5)

## Reward Function
![method_reward_3](https://github.com/user-attachments/assets/93ebd24b-276c-4d0c-8e55-1e83cda12389)

## Seismic Hazard
![method_community_robustness](https://github.com/user-attachments/assets/31bac2b5-2efb-4332-be2a-6bfe2fac7306)

## Fragility Curves
![frag_curves](https://github.com/user-attachments/assets/18a72b28-dffd-4d56-b921-0f5140527b39)


## Methodology
![method_workflow_overview](https://github.com/user-attachments/assets/f3156b0d-3492-45de-b9a7-066cc4227bd4)



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
