import kfp
# Load the component by calling load_component_from_file or load_component_from_url
# To load the component, the pipeline author only needs to have access to the component.yaml file.
# The Kubernetes cluster executing the pipeline needs access to the container image specified in the component.
dummy_op = kfp.components.load_component_from_file(os.path.join(component_root, 'component.yaml')) 
# dummy_op = kfp.components.load_component_from_url('http://....../component.yaml')

## Load two more components for importing and exporting the data:
#download_from_gcs_op = kfp.components.load_component_from_url('http://....../component.yaml')
#upload_to_gcs_op = kfp.components.load_component_from_url('http://....../component.yaml')

# dummy_op is now a "factory function" that accepts the arguments for the component's inputs
# and produces a task object (e.g. ContainerOp instance).
# Inspect the dummy_op function in Jupyter Notebook by typing "dummy_op(" and pressing Shift+Tab
# You can also get help by writing help(dummy_op) or dummy_op? or dummy_op??
# The signature of the dummy_op function corresponds to the inputs section of the component.
# Some tweaks are performed to make the signature valid and pythonic:
# 1) All inputs with default values will come after the inputs without default values
# 2) The input names are converted to pythonic names (spaces and symbols replaced
#    with underscores and letters lowercased).

# Define a pipeline and create a task from a component:
def my_pipeline():
    dummy1_task = dummy_op(
        # Input name "Input 1" is converted to pythonic parameter name "input_1"
        input_1="one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten",
        parameter_1='5',
    )
    # The outputs of the dummy1_task can be referenced using the
    # dummy1_task.outputs dictionary: dummy1_task.outputs['output_1']
    # ! The output names are converted to pythonic ("snake_case") names.

# This pipeline can be compiled, uploaded and submitted for execution.
kfp.Client().create_run_from_pipeline_func(my_pipeline, arguments={})
