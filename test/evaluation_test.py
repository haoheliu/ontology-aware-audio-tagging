import sys
sys.path.append("..")

from ontology_audio_tagging.utils import load_pickle
from ontology_audio_tagging import ontology_mean_average_precision

panns_stats = load_pickle("stats/ast_0.456.pkl")

audio_name, clipwise_output, target = panns_stats['audio_name'], panns_stats['clipwise_output'], panns_stats['target']

evaluation_result = ontology_mean_average_precision(target, clipwise_output)

import ipdb; ipdb.set_trace()