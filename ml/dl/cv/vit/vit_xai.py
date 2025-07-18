# a) SHAP for Multi-Modal Inputs

import shap

# 1) Define masker for each modality
img_masker = shap.maskers.Image("inpaint_telea", input_shape=(3,224,224))
tab_masker = shap.maskers.Independent(X_tab[:100])

# 2) Composite
masker = shap.maskers.Composite([img_masker, tab_masker])

# 3) Wrap model to accept tuple
def model_wrapper(inputs):
    images, tabs = inputs
    logits = model_cls(images, tabs)
    return torch.softmax(logits, dim=1).detach().cpu().numpy()

explainer = shap.Explainer(model_wrapper, masker)
# explain one example
shap_values = explainer((df_images[0:1], X_tab[0:1]))
# visualize
shap.image_plot(shap_values[:,0], df_images[0:1])   # for class 0
shap.plots.bar(shap_values[:,:,1])                  # tabular importances








# b) Attention-Based Heatmaps
# 1) Forward with attentions
outputs = model_cls.vit.forward_features(img, output_attentions=True)
attns = outputs.attentions   # list of [B, heads, N, N]

# 2) Attention rollout (per Abnar & Zuidema, 2020)
def rollout(attns, discard_ratio=0.9):
    result = torch.eye(attns[0].size(-1))
    for attn in attns:
        attn_heads = attn.mean(1)  # average heads
        flat = attn_heads.view(result.size(0), -1)
        _, idx = flat.topk(int(flat.size(1)*discard_ratio), dim=1, largest=False)
        mask = torch.ones_like(flat)
        mask.scatter_(1, idx, 0)
        attn_heads = attn_heads * mask.view_as(attn_heads)
        attn_heads = attn_heads / attn_heads.sum(-1, keepdim=True)
        result = attn_heads @ result
    return result

roll = rollout(attns)
# 3) extract cls-token attention to patches, reshape & upsample to 224×224, overlay on image.


# c) Captum for Layer-Wise Attribution
from captum.attr import IntegratedGradients, LayerGradCam
ig = IntegratedGradients(model_cls)
attributions, delta = ig.attribute((img, tab), target=pred_class, 
                                   return_convergence_delta=True)
# Split outputs:
img_attr, tab_attr = attributions

# Visualize image attributions:
#   convert img_attr to numpy heatmap, overlay on img

# Visualize tabular attributions:
for feat, score in zip(all_tab_feature_names, tab_attr[0].cpu().numpy()):
    print(f"{feat}: {score:.4f}")
