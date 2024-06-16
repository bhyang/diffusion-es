# Diffusion-ES: Gradient-free Planning with Diffusion for Autonomous Driving and Zero-Shot Instruction Following (CVPR 2024)

## Getting started
### Download the nuPlan data
You can install the nuPlan dataset by following instructions from [here](https://github.com/motional/nuplan-devkit/blob/master/docs/dataset_setup.md).
Make sure to set the environment variables pointing to the dataset:
```
export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"
```
### Setting up the environment
Clone this repo:
```
git clone https://github.com/bhyang/diffusion-es.git
cd diffusion-es
```
Install the nuplan-devkit conda environment.
```
cd nuplan-devkit
conda env create -f environment.yml
```
The extra dependencies that aren't in the original nuplan-devkit are in `requirements_diffusiones.txt` and can be installed separately if you already have an existing environment.

Install the local repos as packages:
```
pip install -e nuplan-devkit
pip install -e tuplan_garage
```

Set environment variables to set your output directory and point to the nuplan-devkit:
```
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
export NUPLAN_DEVKIT_ROOT="$HOME/diffusion-es/nuplan-devkit/"
```

## Running the code
### Training diffusion models
Train the unconditional diffusion model:
```
sh scripts/train.sh
```
Train the conditional diffusion model:
```
sh scripts/train_cond.sh
```
To reproduce the main Diffusion-ES results in the paper, use the unconditional diffusion model.

### Running Diffusion-ES on nuPlan val14
Run Diffusion-ES on the val14 benchmark:
```
sh scripts/eval_diffusion_es.sh
```
The script needs to be modified to point to your model checkpoint.

### Running Diffusion-ES for language controllability
To run the few-shot LLM prompting experiments, set the environment variable `OPENAI_API_KEY`. Note that running this may result in charges to your OpenAI account. The code can be modified to not invoke the OpenAI API, but just as a heads up this is the default behavior.

The language command can be specified in the script directly (as is the case for the controllability scripts). For best results, take a look at the examples provided in `tuplan_garage/tuplan_garage/planning/simulation/planner/pdm_planner/language`.

The scripts for running the language controllability experiments are in `scripts/controllability`:
```
sh scripts/controllability/controllability_01_ours.sh
sh scripts/controllability/controllability_02_ours.sh
...
```

## Acknowledgements
This code is largely based off [nuplan-devkit](https://github.com/motional/nuplan-devkit) and [tuplan_garage](https://github.com/bhyang/tuplan_garage). Special thanks to the respective authors for making this work possible!

If you do find this code useful in your own research, you can cite the paper:
```
@article{yang2024diffusion,
  title={Diffusion-ES: Gradient-free Planning with Diffusion for Autonomous Driving and Zero-Shot Instruction Following},
  author={Yang, Brian and Su, Huangyuan and Gkanatsios, Nikolaos and Ke, Tsung-Wei and Jain, Ayush and Schneider, Jeff and Fragkiadaki, Katerina},
  journal={arXiv preprint arXiv:2402.06559},
  year={2024}
}
```
