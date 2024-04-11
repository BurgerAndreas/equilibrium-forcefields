from ocpmodels.tasks.task import BaseTask
from ocpmodels.common.registry import registry


@registry.register_task("compute_stats")
class ComputeStatsTask(BaseTask):
    def run(self):
        # print(f'In ComputeStatsTask.run()')
        self.trainer.compute_stats()
