import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.components as comp
import os

simple_op = comp.load_component_from_file(os.path.join(
    "./components/simple/", 'simple_component.yaml'))

train_op = comp.load_component_from_file(os.path.join(
    "./components/train/", 'train_component.yaml'))

@dsl.pipeline(
    name='Simple Pipeline',
    description='Simple pipeline with a simple workload'
)
# Define a pipeline and create a task from a component
def simple_pipeline(
    pipeline_param: 'String' = 'pipeline param'
):
    simple1_task = simple_op(
        input_1="one\ntwo\nthree\nfour\nfive\nsix\nseven",
        parameter_1='5'
    )
    train_task = train_op(
        input_1=simple1_task.outputs['output_1'],
    ).set_gpu_limit(1).add_node_selector_constraint(
            'cloud.google.com/gke-accelerator', 'nvidia-tesla-t4')

# generate compressed pipeline file for upload
if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(simple_pipeline, __file__ + '.tar.gz')
