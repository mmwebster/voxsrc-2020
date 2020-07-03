import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.components as comp
import os

preproc_op = comp.load_component_from_file(os.path.join(
    "./components/preproc/", 'preproc_component.yaml'))

train_op = comp.load_component_from_file(os.path.join(
    "./components/train/", 'train_component.yaml'))

@dsl.pipeline(
    name='VoxCeleb Baseline Reproduction Pipeline',
    description='Train baseline models'
)
# Define a pipeline and create a task from a component
def baseline_repro_pipeline(
    data_bucket: str = 'voxsrc-2020-voxceleb-v4',
    test_list: str = 'vox1_no_cuda.txt',
    train_list: str = 'vox2_no_cuda.txt',
    test_path: str = 'vox1_no_cuda.tar.gz',
    train_path: str = 'vox2_no_cuda.tar.gz',
    batch_size: int = 5,
    max_epoch: int = 1,
):
    preproc_task = preproc_op(
        data_bucket = data_bucket,
        test_list = test_list,
        train_list = train_list,
        test_path = test_path,
        train_path = train_path,
    )

    train_task = train_op(
        data_bucket = preproc_task.outputs['data_bucket'],
        test_list = test_list,
        train_list = train_list,
        test_path = test_path,
        train_path = train_path,
        batch_size = batch_size,
        max_epoch = max_epoch,
    ).apply(gcp.use_preemptible_nodepool(hard_constraint=True))\
     .set_gpu_limit(1)\
     .add_node_selector_constraint('cloud.google.com/gke-accelerator',
             'nvidia-tesla-t4')

     train_task.after(preproc_task)

# generate compressed pipeline file for upload
if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(baseline_repro_pipeline, __file__ + '.tar.gz')
