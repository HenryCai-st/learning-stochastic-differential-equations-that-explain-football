# Deep Learning Project

## Project description

We are to achieve "trajectories to parameters", the inverse design of simulating differential equation using parameters based on "Learning Stochastic Differential Equations that Explain Football".

For the current, following steps are planned:

1. **Reproduce Simulation**

Know how we generate trajectories based on parameters. This is assumed to be provided.

2. **Modelling**

Model the whole piepline of project. e.g.:
- Design parameters for predicting the trajectories
- Decide shape of dataset, 2d with trajectory length, type image or sequence of coordinates
- Decide architecture used for prediction
- Parameter sensitivity analysis: find out how parameters influence distribution of SDEs

3. **Prototype**

- Start with simple CNN, we can set up binary classifier first to let it separate one distribution of traectories to another
- Note: Need to find parameters with maximal difference in distribution regarding parameter sensitivity analysis
- Then if result looks good, use transfer learning to finetune to do regression on parameters

4. **Iteration**

- After prototype, one evaluates using real generated simulation and simulation based on predicted parameters
- Then try use larger dataset by ust generating more simulations, more advanced architecture, additionals techniques

5. **Report**

- This should begin as soon as the pipeline proceeds
- Take notes on every iteration, modelling
- We can separate different iterations by branching
- Also note down expermient and model design every iteration

## Setup Instructions

1. **Create and Activate a Virtual Environment:**
   Using `venv`:
   ```bash
   python -m venv venv
   ```
   * Windows: `venv\Scripts\activate`
   * Mac/Linux: `source venv/bin/activate`

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch with CUDA 12.8:**
   If you have the CPU version installed, uninstall it first:
   ```bash
   pip uninstall torch torchvision
   ```
   Then install the CUDA-enabled version:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
