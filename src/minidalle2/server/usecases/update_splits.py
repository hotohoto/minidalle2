from minidalle2.server.repositories.annotation_repository import AnnotationRepository


def update_splits(
    annotation_repo: AnnotationRepository,
):
    annotation_repo.reset_indices()
