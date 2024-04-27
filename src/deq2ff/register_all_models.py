# register models to be used with EquiformerV1 training loop (MD17)
from equiformer.nets.registry import register_model

# DEQ EquiformerV2
from deq2ff.deq_equiformer_v2.deq_equiformer_v2_oc20 import DEQ_EquiformerV2_OC20


@register_model
def deq_equiformer_v2_oc20(**kwargs):
    return DEQ_EquiformerV2_OC20(**kwargs)


from deq2ff.deq_equiformer_v2.deq_equiformer_v2_md17 import DEQ_EquiformerV2_MD17
@register_model
def deq_equiformer_v2_oc20(**kwargs):
    return DEQ_EquiformerV2_MD17(**kwargs)

# EquiformerV2
from equiformer_v2.nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20


@register_model
def equiformer_v2_oc20(**kwargs):
    return EquiformerV2_OC20(**kwargs)


from equiformer_v2.nets.equiformer_v2.equiformer_v2_md17 import EquiformerV2_MD17


@register_model
def equiformer_v2_md17(**kwargs):
    return EquiformerV2_MD17(**kwargs)