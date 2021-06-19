import os

from azureml.core import Workspace, Environment, Experiment, Model, ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies

# Load the workspace from the saved config file            
ws = Workspace.from_config()
experiment_folder = '/home/azureuser/cloudfiles/code/Users/s202581/project/first_experiment'
log_folder = '/home/azureuser/cloudfiles/code/Users/s202581/project/first_experiment/run_downloaded_logs'
target = ComputeTarget(workspace=ws, name ='MLOps-GPU')

# Create a Python environment for the experiment
env = Environment.from_pip_requirements('env', '/home/azureuser/cloudfiles/code/Users/s202581/project/first_experiment/requirements.txt')

# # Ensure the required packages are installed (we need pip and Azure ML defaults)
# packages = CondaDependencies.create(conda_packages=['pip','pytorch','torchvision','matplotlib','joblib'],
#                                     pip_packages=['azureml-defaults'])
# env.python.conda_dependencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory=experiment_folder,
                                script='src/models/train_model.py',
                                compute_target=target,
                                environment=env)

# Create an Azure ML experiment in your workspace
experiment = Experiment(workspace=ws, name="train_test_model")
run = experiment.submit(config=script_config)
print("Starting experiment:", experiment.name)

run.wait_for_completion()

# Download all files
run.get_all_logs(destination=log_folder)

# MAKE SURE NEXT TIME YOU SAVE THE MODEL IN A LOCAL FOLDER SO THAT YOU CAN ACCESS THE .PKL FILES LATER

# Verify the files have been downloaded
for root, directories, filenames in os.walk(log_folder): 
    for filename in filenames:  
        print (os.path.join(root,filename))

