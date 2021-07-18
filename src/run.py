from omegaconf import OmegaConf
from work import main

################################################
CFG = OmegaConf.load(
    "/workspaces/commonlitreadabilityprize/config/config.yaml"
)


CFG.general.update(
    {
        "seed": 9,
    }
)
CFG.scheduler.params.update(
    {
        "max_lr": 5e-05
    }
)

main(CFG)
################################################

CFG = OmegaConf.load(
    "/workspaces/commonlitreadabilityprize/config/config.yaml"
)

CFG.general.update(
    {
        "seed": 10,
    }
)
CFG.optimizer.params.update(
    {
        "weight_decay": 0.0
    }
)
CFG.scheduler.params.update(
    {
        "max_lr": 5e-05
    }
)
main(CFG)

################################################

CFG = OmegaConf.load(
    "/workspaces/commonlitreadabilityprize/config/config.yaml"
)

CFG.general.update(
    {
        "seed": 11,
    }
)
CFG.optimizer.params.update(
    {
        "weight_decay": 0.0
    }
)
CFG.scheduler.params.update(
    {
        "max_lr": 5e-05
    }
)
CFG.training.update({"stochastic_weight_avg": True})
main(CFG)

################################################


CFG = OmegaConf.load(
    "/workspaces/commonlitreadabilityprize/config/config.yaml"
)

main(CFG)
