from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.diversity_filters.molformer.identical_murcko_scaffold import \
    IdenticalMurckoScaffold
from reinvent_scoring.scoring.diversity_filters.molformer.no_filter import NoFilter
from reinvent_scoring.scoring.diversity_filters.molformer.no_filter_with_penalty import NoFilterWithPenalty


class DiversityFilter:

    def __new__(cls, parameters: DiversityFilterParameters) -> BaseDiversityFilter:
        all_filters = dict(NoFilterWithPenalty=NoFilterWithPenalty,
                           NoFilter=NoFilter,
                           IdenticalMurckoScaffold=IdenticalMurckoScaffold
                           )
        div_filter = all_filters.get(parameters.name, KeyError(f"Invalid filter name: `{parameters.name}'"))
        return div_filter(parameters)
