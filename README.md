# ONTOLOGY-AWARE LEARNING AND EVALUATION FOR AUDIO TAGGING

Code supplimentary for our ICASSP 2023 submission

This study defines a new evaluation metric for audio tagging tasks to overcome the limitation of the conventional mean average precision (mAP) metric, which treats different kinds of sound as independent classes without considering their relations. Also, due to the ambiguities in sound labeling, the labels in the training and evaluation set are not guaranteed to be accurate and exhaustive, which poses challenges for robust evaluation with mAP. The proposed metric, ontology-aware mean average precision (OmAP) addresses the weaknesses of mAP by utilizing the AudioSet ontology information during the evaluation. Specifically, we reweight the false positive events in the model prediction based on the ontology graph distance to the target classes. The OmAP measure also provides more insights into model performance by evaluations with different coarse-grained levels in the ontology graph. We conduct human evaluations and demonstrate that OmAP is more consistent with human perception than mAP. To further verify the importance of utilizing the ontology information, we also propose a novel loss function (OBCE) that reweights binary cross entropy (BCE) loss based on the ontology distance. Our experiment shows that OBCE can improve both mAP and OmAP metrics on the AudioSet tagging task. 

---

I'm organizing the code. Coming soon.