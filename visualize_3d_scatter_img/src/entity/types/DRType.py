from enum import Enum


class DRType(Enum):
    ISOMAP = 'ISOMAP'
    TSNE = 'TSNE'
    UMAP = 'UMAP'
    PCA = 'PCA'
    MDS = 'MDS'
