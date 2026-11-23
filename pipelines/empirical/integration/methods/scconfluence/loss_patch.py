"""GPL-3.0-only: corrected accumulation for pinned scConfluence ScConfluence.loss.

Adapted from cantinilab/scconfluence@14d81e2ca679e161ec5c6806103a66c14fcdcd9a.
The native reconstruction, partial-OT, Sinkhorn and weighting functions are retained.
"""

import torch


def corrected_loss(self, x, inference_output, generative_output, reduce=True):
    losses = {}
    for modality in self.modalities:
        if not len(x[modality]["input"]):
            continue  # A mixed minibatch may contain no observations from one assay.
        values = self.aes[modality].loss(
            x[modality], inference_output=inference_output[modality],
            generative_output=generative_output[modality], reduce=reduce)
        losses.update({key + "_" + modality: value for key, value in values.items()})
        self.metrics_to_log += self.aes[modality].metrics_to_log
    iot_sum = x[self.modalities[0]]["input"].new_zeros(1)
    sinkhorn_sum = iot_sum.clone()
    for pair in self.modality_pairs:
        first, second = pair.split("+")
        z1, z2 = inference_output[first]["z"], inference_output[second]["z"]
        if len(z1) * len(z2):
            iot = self.get_iot_loss(z_1=z1, z_2=z2, c_cross=x["cross"][pair])
            sinkhorn = self.get_sinkhorn_reg(z_1=z1, z_2=z2)
            for name, value in (("iot_loss_", iot), ("sinkhorn_reg_", sinkhorn)):
                self.metrics_to_log.append({"name": name + pair, "value": value,
                                            "batch_size": 1, "reduce_fx": torch.mean})
            iot_sum = iot_sum + iot
            sinkhorn_sum = sinkhorn_sum + sinkhorn
    losses["iot_loss"] = self.iot_loss_weight * iot_sum
    losses["sinkhorn_reg"] = self.sinkhorn_loss_weight * sinkhorn_sum
    return losses


def nonempty_latent_logs(self, inference_dic, return_log=False):
    for modality in self.modalities:
        if len(inference_dic[modality]["z"]):
            self.metrics_to_log.append(self.aes[modality].log_latent_norms(
                inference_dic[modality], return_log=True))
