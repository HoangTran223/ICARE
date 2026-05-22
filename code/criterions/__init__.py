from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .MCW_KD import MCW_KD
from .MCW_KD_Dual import MCW_KD_Dual
from .DSKD import DualSpaceKDWithCMA
from .DSKD_IMPACT import DualSpaceKDWithIMPACT
from .ULD import UniversalLogitDistillation
from .MinED import MinEditDisForwardKLD
from .MultiLevelOT import MultiLevelOT
from .SRA import SRA
from .ALM import ALM
from .ResidualKD import ResidualKD
from .IMPACT import IMPACT
from .SRA_IMPACT import SRA_IMPACT
from .ALM_IMPACT import ALM_IMPACT
from .ResidualKD_IMPACT import ResidualKD_IMPACT

criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "dual_space_kd_with_cma": DualSpaceKDWithCMA,
    "DSKD_IMPACT": DualSpaceKDWithIMPACT,
    "dual_space_kd_with_cma_impact": DualSpaceKDWithIMPACT,
    "universal_logit_distillation": UniversalLogitDistillation,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "MCW_KD": MCW_KD,
    "MCW_KD_Dual": MCW_KD_Dual,
    "MultiLevelOT": MultiLevelOT,
    "SRA": SRA,
    "ALM": ALM,
    "ResidualKD": ResidualKD,
    "IMPACT": IMPACT,
    "SRA_IMPACT": SRA_IMPACT,
    "ALM_IMPACT": ALM_IMPACT,
    "ResidualKD_IMPACT": ResidualKD_IMPACT,
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")