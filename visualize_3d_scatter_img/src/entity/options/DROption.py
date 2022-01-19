import dataclasses
from src.entity.types.DRType import DRType


class DROption:
    def __init__(self, dr_type=DRType.ISOMAP, **kwargs):
        self.dr_type = dr_type
        if self.dr_type == DRType.ISOMAP:
            self.argments = IsomapOption(components=kwargs['components'])
        elif self.dr_type == DRType.TSNE:
            self.argments = TsneOption(components=kwargs['components'])
        elif self.dr_type == DRType.PCA:
            self.argments = PcaOption(components=kwargs['components'])
        elif self.dr_type == DRType.UMAP:
            self.argments = UmapOption(components=kwargs['components'])
        elif self.dr_type == DRType.MDS:
            self.argments = MdsOption(components=kwargs['components'])


@dataclasses.dataclass(frozen=True)
class IsomapOption:
    components: int = 3
    n_neighbors: int = 5
    neighbors_algorithm: str = 'brute'


@dataclasses.dataclass(frozen=True)
class TsneOption:
    components: int = 3
    perplexity: int = 100
    n_iter: int = 1000
    learning_rate: int = 1000
    init: str = 'random'


@dataclasses.dataclass(frozen=True)
class UmapOption:
    components: int = 3
    n_neighbors: int = 10
    min_dist: float = 1.0


@dataclasses.dataclass(frozen=True)
class PcaOption:
    components: int = 3
    whiten: bool = True


@dataclasses.dataclass(frozen=True)
class MdsOption:
    components: int = 3
    metric: bool = True
    dissimilarity: str = 'euclidean'
