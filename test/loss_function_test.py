import torch
from ontology_audio_tagging import ontology_binary_cross_entropy

batchsize = 32
class_num_audioset = 527

target = torch.randn((batchsize, class_num_audioset)).cuda()
output = torch.randn((batchsize, class_num_audioset)).cuda()

loss = ontology_binary_cross_entropy(output, target)

print("Value of the ontology-based binary cross entropy loss: ", loss)
