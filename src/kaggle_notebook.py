from omegaconf import OmegaConf
from work import main
CFG = OmegaConf.load(
    "/workspaces/commonlitreadabilityprize/config/config.yaml"
)


CFG.general.update(
    {
        "seed": 7,
    }
)
CFG.loss.params.update(
    {
        "beta": 0.1,
    }
)

main(CFG)


CFG = OmegaConf.load(
    "/workspaces/commonlitreadabilityprize/config/config.yaml"
)

main(CFG)
