import torch


def parameters_to_vector(model: torch.nn.Module):
    """Given a torch model returns a torch tensor with all its parameters"""
    return torch.cat([p.data.view(-1) for p in model.parameters()], dim=0)


def convert(models):
    """Converts a list of models into a list of vectors, each vector represents
    the flattened parameters of the corresponding model"""
    params = []
    for model in models:
        if isinstance(model, torch.nn.Module):
            model = parameters_to_vector(model)

        params.append(model)

    return params


def dot(x, y):
    """Dot product normalized by the number of elements"""
    x, y = convert([x, y])
    return x @ y / x.numel()


def norm(x):
    """L2 norm normalized by the number of elements"""
    return torch.sqrt(dot(x, x))


def cosine_similarity(x, y):
    """Cosine similarity between two vectors"""
    return dot(x, y) / (norm(x) * norm(y))


def distance(x, y):
    """Distance between two vectors normalized by the number of elements"""
    x, y = convert([x, y])
    return norm(x - y)


def normalized_distance(x, y):
    """Distance between two vectors normalized by the number of elements and
     the norm of the first vector"""
    return distance(x, y) / torch.sqrt(norm(x) * norm(y))


def distance_matrix(vectors, distance_fn=cosine_similarity):
    """Computes the distance matrix between all the vectors given a distance"""
    vectors = convert(vectors)
    n_vectors = len(vectors)
    matrix = torch.empty((n_vectors, n_vectors))

    for i in range(n_vectors):
        for j in range(n_vectors):
            matrix[i, j] = distance_fn(vectors[i], vectors[j])

    return matrix
