import os

import azureml
from azureml.core import Workspace, Environment, Experiment, Model, ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies

azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 2000000000

# Load the workspace from the saved config file            
ws = Workspace.from_config()
experiment_folder = '/home/azureuser/cloudfiles/code/Users/s202581/MLOP_final_project/'
log_folder = experiment_folder + 'run_downloaded_logs/'
target = ComputeTarget(workspace=ws, name ='MLOps-BigBoi')

# Create a Python environment for the experiment
env = Environment.from_pip_requirements('env', '/home/azureuser/cloudfiles/code/Users/s202581/MLOP_final_project/requirements.txt')

# Create a script config
script_config = ScriptRunConfig(source_directory=experiment_folder,
                                script='/home/azureuser/cloudfiles/code/Users/s202581/MLOP_final_project/src/models/train_model.py',
                                arguments=['-dataset','augmented','-batch_size',4, '-no_epochs', 1],
                                compute_target=target,
                                environment=env)

# Create an Azure ML experiment in your workspace
experiment = Experiment(workspace=ws, name="train_test_sheep")
run = experiment.submit(config=script_config)
print("Starting experiment:", experiment.name)

run.wait_for_completion()

# Download all files
run.get_all_logs(destination=log_folder)

# Verify the files have been downloaded
for root, directories, filenames in os.walk(log_folder): 
    for filename in filenames:  
        print (os.path.join(root,filename))

