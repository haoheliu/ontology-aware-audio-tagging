# ONTOLOGY-AWARE LEARNING AND EVALUATION FOR AUDIO TAGGING

Code supplimentary for our ICASSP 2023 submission

I'm organizing the code. Coming soon.

---

This study defines a new evaluation metric for audio tagging tasks to overcome the limitation of the conventional mean average precision (mAP) metric, which treats different kinds of sound as independent classes without considering their relations. Also, due to the ambiguities in sound labeling, the labels in the training and evaluation set are not guaranteed to be accurate and exhaustive, which poses challenges for robust evaluation with mAP. The proposed metric, ontology-aware mean average precision (OmAP) addresses the weaknesses of mAP by utilizing the AudioSet ontology information during the evaluation. Specifically, we reweight the false positive events in the model prediction based on the ontology graph distance to the target classes. The OmAP measure also provides more insights into model performance by evaluations with different coarse-grained levels in the ontology graph. We conduct human evaluations and demonstrate that OmAP is more consistent with human perception than mAP. To further verify the importance of utilizing the ontology information, we also propose a novel loss function (OBCE) that reweights binary cross entropy (BCE) loss based on the ontology distance. Our experiment shows that OBCE can improve both mAP and OmAP metrics on the AudioSet tagging task. 

---

# Installation

```shell
git clone git@github.com:haoheliu/ontology-aware-audio-tagging.git
cd ontology-aware-audio-tagging
pip3 install -e .
```


# Ontology-aware evaluation matric for audio tagging (OmAP)

Please clone and run the following code. 

```python
# test/evaluation_metric_test.py
from ontology_audio_tagging.utils import load_pickle
from ontology_audio_tagging import ontology_mean_average_precision

panns_stats = load_pickle("stats/ast_0.456.pkl")

audio_name, clipwise_output, target = (
    panns_stats["audio_name"],
    panns_stats["clipwise_output"],
    panns_stats["target"],
)

(   omap_average, # Float. The average OmAP at different coarse level.
    omap_on_different_coarse_level, # A dict object. The OmAP (value) at different coarse levels (key).
    omap_on_different_coarse_level_details, # A dict object. The OmAP for each class (value) at different coarse levels (key).
) = ontology_mean_average_precision(target, clipwise_output)

print("The average ontology mAP of AST model is %.3f." % omap_average)
print("The ontology mAP of AST model at different coarse levels are %s" % omap_on_different_coarse_level)

```

# Ontology aware loss function for audio tagging (OBCE)

Please clone and run the following code. You can use OBCE in your audio tagging model training.

```python
# test/loss_function_test.py

import torch
from ontology_audio_tagging import ontology_binary_cross_entropy

batchsize = 32
class_num_audioset = 527

target = torch.randn((batchsize, class_num_audioset)).cuda()
output = torch.randn((batchsize, class_num_audioset)).cuda()

loss = ontology_binary_cross_entropy(output, target) # Use this loss for back propagation

print("Value of the ontology-based binary cross entropy loss: ", loss)

```