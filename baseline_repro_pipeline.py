import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.components as comp
import os

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
    checkpoint_bucket: str = 'voxsrc-2020-checkpoints',
    batch_size: int = 5,
    max_epoch: int = 1,
):
    use_preemptible = False
    use_gpu = False
    run_id = '{{workflow.uid}}'

    train_task = train_op(
        data_bucket = data_bucket,
        test_list = test_list,
        train_list = train_list,
        test_path = test_path,
        train_path = train_path,
        batch_size = batch_size,
        max_epoch = max_epoch,
        checkpoint_bucket = checkpoint_bucket,
        run_id = run_id,
    )

    # @brief Require training to run on a preemtible node pool
    # @note This autoscales an autoscalable node pool from 0->1 that
    #       matches the corresponding config. Autoscaled nodes will be
    #       deactivated on GCP after 10 minutes of inactivity
    if use_preemptible:
        train_task\
            .apply(gcp.use_preemptible_nodepool(hard_constraint=True))\
            .set_retry(5)

    # @brief Select only a node pool with 1 Nvidia Tesla T4
    if use_gpu:
        train_task\
            .set_gpu_limit(1)\
            .add_node_selector_constraint('cloud.google.com/gke-accelerator',
                    'nvidia-tesla-t4')

# generate compressed pipeline file for upload
if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(baseline_repro_pipeline, __file__ + '.tar.gz')
