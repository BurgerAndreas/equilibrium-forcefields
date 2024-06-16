import wandb
from tqdm import tqdm

# https://wandb.ai/andreas-burger/EquilibriumEquiFormer
project = "EquilibriumEquiFormer"
author = "andreas-burger"

# get all runs where config model_is_deq:true and is finished, failed, or crashed
api = wandb.Api()
# runs = api.runs(f"{author}/{project}", {"config.model_is_deq": True})
runs = api.runs(
    f"{author}/{project}", 
    {"$or": [{"state": "finished"}, {"state": "crashed"}, {"state": "failed"}]}
)


# rename all runs
for run in tqdm(runs):
    _name = run.name
    new_name = run.name
    # get config deq_kwargs.f_tol
    try:
        f_tol = run.config["deq_kwargs"]["f_tol"]

        if f_tol == 1e-2:
            new_name = new_name.replace("ar ", "ar2 ")
            new_name = new_name.replace("ard ", "ar2d ")
            new_name = new_name.replace("ard2 ", "ar2d2 ")
            new_name = new_name.replace("ardnull ", "ar2dnull ")
            new_name = new_name.replace("brd ", "br2d ")
            new_name = new_name.replace("brdnull ", "br2dnull ")
        
    except KeyError:
        print(f"run {_name} has no f_tol.")
        continue

    new_name = new_name.replace("target-", "")

    if _name != new_name:
        print(f"{_name}  ->  {new_name}")

        # DANGER

        # # does not work
        # run.name = new_name
        # run.update()

        # works
        # https://github.com/wandb/wandb/issues/3684
        run.display_name = new_name
        run.update()

        # # reinit run take too long
        # wandb.init(project=project, entity=author, id=run.id, resume="must")
        # wandb.run.name = new_name
        # wandb.run.finish()

        # DANGER

    else:
        print(f"(unchanged) {_name}")