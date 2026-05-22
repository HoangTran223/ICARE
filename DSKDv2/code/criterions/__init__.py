from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .dual_space_kd_v2_with_exact_token_alignment import DualSpaceKDV2WithETA
from .dskdv2_impact import DualSpaceKDV2WithIMPACT

criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "dual_space_kd_v2_with_eta": DualSpaceKDV2WithETA,
    "dual_space_kd_v2_impact": DualSpaceKDV2WithIMPACT,
    # Aliases kept for older checkpoint args.json
    "dual_space_kd_v2_ipact": DualSpaceKDV2WithIMPACT,
}


def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    raise NameError(f"Undefined criterion for {args.criterion}!")
