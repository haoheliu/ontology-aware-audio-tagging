from ontology_audio_tagging.utils import load_pickle
from ontology_audio_tagging import ontology_mean_average_precision

panns_stats = load_pickle("stats/ast_0.456.pkl")

audio_name, clipwise_output, target = (
    panns_stats["audio_name"],
    panns_stats["clipwise_output"],
    panns_stats["target"],
)

(
    omap_average,
    omap_on_different_coarse_level,
    omap_on_different_coarse_level_details,
) = ontology_mean_average_precision(target, clipwise_output)

print("The average ontology mAP of AST model is %.3f." % omap_average)
print(
    "The ontology mAP of AST model at different coarse levels are %s"
    % omap_on_different_coarse_level
)
