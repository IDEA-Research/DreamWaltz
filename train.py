import pyrallis
from configs import TrainConfig
from core.trainer import Trainer
from core.pretrainer import PreTrainer


@pyrallis.wrap()
def main(cfg: TrainConfig):

    if cfg.log.pretrain_only:
        trainer = PreTrainer(cfg)
    else:
        trainer = Trainer(cfg)

    if cfg.log.eval_only:
        trainer.full_eval()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
