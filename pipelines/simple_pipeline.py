import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.components as comp
import os

component_root = "../"
simple_op = comp.load_component_from_file(os.path.join(
    component_root, 'simple_component.yaml')) 
# dummy_op =comp.load_component_from_url('http://....../component.yaml')

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
    simple2_task = simple_op(
        input_1=simple1_task.outputs['output_1'],
        parameter_1='3'
    )
    simple3_task = simple_op(
        input_1=simple2_task.outputs['output_1'],
        parameter_1='1'
    )

# generate compressed pipeline file for upload
if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(simple_pipeline, __file__ + '.tar.gz')
