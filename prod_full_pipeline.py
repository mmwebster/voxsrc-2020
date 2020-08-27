import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.components as comp
import os
from kubernetes import client as k8s_client

feature_extraction_op = comp.load_component_from_file(os.path.join(
    "./components/feature-extractor/", 'feature_extractor_component.yaml'))

ipc_shared_mem_volume = dsl.PipelineVolume(name='shm-vol', empty_dir={'medium': 'Memory'})

train_op = comp.load_component_from_file(os.path.join(
    "./components/train/", 'train_component.yaml'))

@dsl.pipeline(
    name='VoxCeleb Baseline Reproduction Pipeline',
    description='Train baseline models'
)
# Define a pipeline and create a task from a component
# @TODO abstract code shared with test_full_pipeline.py
def baseline_repro_pipeline(
    data_bucket: str = 'voxsrc-2020-voxceleb-v4',
    test_list: str = 'vox1_full.txt',
    # @note test_utterances_list is in the same format as train_list, but for
    #       the test data. Whereas test_list contains utterance pairs for
    #       evaluation
    test_utterances_list: str = 'vox1_full_utterances.txt',
    train_list: str = 'vox2_full.txt',
    test_path: str = 'vox1_full.tar.gz',
    train_path: str = 'vox2_full.tar.gz',
    checkpoint_bucket: str = 'voxsrc-2020-checkpoints',
    batch_size: int = 750,
    max_epoch: int = 21,
    n_speakers: int = 2,
    test_interval: int = 3,
    feature_extraction_threads: int = 16,
    data_loader_threads: int = 7,
    # @note This run ID contains "full" pre-extracted features for vox1 and vox2
    reuse_run_with_id: str = "milo_webster-19rvuxfu",
    gaussian_noise_std: float = .9,
):
    # set prod_hw=True to enable production hardware (preemptible V100).
    # Encountered odd issues when node resource constraints aren't known at
    # "compile time" of kf pipeline file
    prod_hw = True
    run_id = '{{workflow.uid}}'

    feature_extraction_task = feature_extraction_op(
        data_bucket = data_bucket,
        test_utterances_list = test_list,
        train_list = train_list,
        test_path = test_path,
        train_path = train_path,
        run_id = run_id,
        num_threads = feature_extraction_threads,
        reuse_run_with_id = reuse_run_with_id
    )

    # default feature extractor to high-perf pool if not in pass-through mode
    # if in pass-through mode, there's no reason to use a beefy node
    if not reuse_run_with_id:
        feature_extraction_task.set_cpu_request("9").set_cpu_limit("16")

    train_task = train_op(
        data_bucket = data_bucket,
        test_list = test_list,
        train_list = train_list,
        test_path = feature_extraction_task.outputs['test_feats_tar_path'],
        train_path = feature_extraction_task.outputs['train_feats_tar_path'],
        batch_size = batch_size,
        max_epoch = max_epoch,
        checkpoint_bucket = checkpoint_bucket,
        run_id = run_id,
        n_speakers = n_speakers,
        test_interval = test_interval,
        gaussian_noise_std = gaussian_noise_std,
        n_data_loader_thread = data_loader_threads,
    )

    train_task.add_pvolumes({'/dev/shm': ipc_shared_mem_volume})
    train_task.after(feature_extraction_task)

    # add Weights & Biases credentials
    if "WANDB_API_KEY" in os.environ:
        train_task.add_env_variable(k8s_client.V1EnvVar(name='WANDB_API_KEY',
            value=os.environ["WANDB_API_KEY"]))
    else:
        raise 'Error: No WandB API key set in environment'

    # @note These resource requests autoscale an autoscalable node pool from
    #       0->1 that matches the corresponding config. Autoscaled nodes will be
    #       deactivated on GCP after 10 minutes of inactivity
    if prod_hw:
        # require training to run on a preemptible node pool
        train_task\
            .apply(gcp.use_preemptible_nodepool(hard_constraint=True))\
            .set_retry(5)
        # require training to run on a node with a gpu of type 'train_gpu_type'
        train_task\
            .set_gpu_limit(1)\
            .add_node_selector_constraint('cloud.google.com/gke-accelerator',
                    'nvidia-tesla-v100')

# generate compressed pipeline file for upload
if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(baseline_repro_pipeline, __file__ + '.tar.gz')
