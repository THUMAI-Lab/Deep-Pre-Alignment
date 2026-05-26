from opencompass.partitioners import NaivePartitioner, InferTimePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

# new infer setting, using split strategy, may speed up the infer time
infer = dict(
    partitioner=dict(
        type=InferTimePartitioner,
        max_task_time=3600,
        strategy='split'),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        max_workers_per_gpu=1,
        task=dict(type=OpenICLInferTask)
    ),
)

infer_num_worker = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        max_workers_per_gpu=1,
        task=dict(type=OpenICLInferTask)
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        max_workers_per_gpu=1,
        task=dict(type=OpenICLEvalTask)),
)
