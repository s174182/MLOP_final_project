from azureml.core import Workspace, Environment, Model
from azureml.core.conda_dependencies import CondaDependencies 

import os

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Load the model and check the version
model = ws.models['sheep_train_augmented']
print(model.name, 'version', model.version)

#Add the script file
script_file = "score.py"

# Add the dependencies for our model (AzureML defaults is already included)
myenv = CondaDependencies()
myenv.add_conda_package('pytorch')
myenv.add_conda_package('torchvision')
myenv.add_conda_package('opencv')

# Save the environment config as a .yml file
env_file = "sheep_deployment_env.yml"
with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
print("Saved dependency info in", env_file)

# Print the .yml file
with open(env_file,"r") as f:
    print(f.read())

from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

# Configure the scoring environment
inference_config = InferenceConfig(runtime= "python",
                                   entry_script=script_file,
                                   conda_file=env_file)

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 5)

service_name = "sheep-service"

service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)
print("we are happy so far")
service.wait_for_deployment(True)
print(service.state)

print(service.get_logs())

endpoint = service.scoring_uri
print('Endpoint: ', endpoint)