import torch


def opacity_loss(pred_ws):
    loss_opacity = torch.sqrt((pred_ws ** 2 + 0.01).mean())
    return loss_opacity


def entropy_loss(pred_ws):
    alphas = pred_ws.clamp(1e-5, 1 - 1e-5)
    entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
    return entropy


def emptiness_loss(pred_ws, emptiness_weight=10000, emptiness_scale=10):
    loss = torch.log(1 + emptiness_scale * pred_ws).mean()
    return emptiness_weight * loss


class SparsityLoss:
    def __init__(self, cfg, use_schedule: bool = True) -> None:
        self.lambda_opacity = cfg.lambda_opacity
        self.lambda_entropy = cfg.lambda_entropy
        self.lambda_emptiness = cfg.lambda_emptiness
        self.use_schedule = use_schedule
        self.sparsity_multiplier = cfg.sparsity_multiplier
        self.sparsity_step = cfg.sparsity_step

    def __call__(self, pred_ws, current_step=None, max_iteration=None):

        loss = 0.0

        if self.lambda_opacity > 0.0:
            loss += self.lambda_opacity * opacity_loss(pred_ws)

        if self.lambda_entropy > 0.0:
            loss += self.lambda_opacity * entropy_loss(pred_ws)

        if self.lambda_emptiness > 0.0:
            loss += self.lambda_emptiness * emptiness_loss(pred_ws)

        if self.use_schedule and current_step >= self.sparsity_step * max_iteration:
            loss *= self.sparsity_multiplier

        return loss
