from omegaconf import OmegaConf
from work import main
CFG = OmegaConf.load(
    "/workspaces/commonlitreadabilityprize/config/config.yaml"
)


CFG.general.update(
    {
        "seed": 101,
    }
)
CFG.optimizer.params.update(
    {
        "lr": 2e-05
    }
)
CFG.scheduler.params.update(
    {
        "max_lr": 1e-03
    }
)

main(CFG)


CFG = OmegaConf.load(
    "/workspaces/commonlitreadabilityprize/config/config.yaml"
)

main(CFG)
