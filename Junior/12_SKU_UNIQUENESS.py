"""SKU UNIQUENESS   
Проблема:
    Рекомендательная система делает подборку товаров для пользователей, но часто эти товары являются 
    однотипными и реализующими одну и ту же потребность пользователя, а это плохо.
Решение:
    Идея такая: прежде чем показывать подборку пользователям, давайте проверим насколько товары в подборке разнообразны.
    Этот web-сервис как раз выполняет эту проверку. 
    Если ответ от сервиса утвердительный, то показываем товары, если нет то рекомендательная система генерирует новую подборку."""

from typing import Tuple
import os
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from sklearn.neighbors import KernelDensity


DIVERSITY_THRESHOLD = 10

app = FastAPI()
embeddings = {}


@app.on_event("startup")
@repeat_every(seconds=10)
def load_embeddings() -> dict:
    """Load embeddings from file."""

    # Load new embeddings each 10 seconds
    path = os.path.join(os.path.dirname(__file__), "embeddings.npy")
    embeddings_raw = np.load(path, allow_pickle=True).item()
    for item_id, embedding in embeddings_raw.items():
        embeddings[item_id] = embedding

    return {}


@app.get("/uniqueness/")
def uniqueness(item_ids: str) -> dict:
    """Calculate uniqueness of each product"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Default answer
    item_uniqueness = {item_id: 0.0 for item_id in item_ids}

    # Calculate uniqueness
    item_embeddings = []
    for item_id in item_ids:
        item_embeddings.append(embeddings[item_id])

    item_embeddings = np.array(item_embeddings)
    item_uniqueness = dict(zip(item_ids, kde_uniqueness(item_embeddings)))
    return item_uniqueness


@app.get("/diversity/")
def diversity(item_ids: str) -> dict:
    """Calculate diversity of group of products"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Default answer
    answer = {"diversity": 0.0, "reject": True}

    # Calculate diversity
    item_embeddings = np.array([embeddings[i] for i in item_ids])  
    reject, mean_diversity = group_diversity(item_embeddings, DIVERSITY_THRESHOLD)
    answer['diversity'] = mean_diversity
    answer['reject'] = reject

    return answer


def kde_uniqueness(product_embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    # Fit a kernel density estimator to the item embedding space
    kde = KernelDensity().fit(product_embeddings)

    uniqueness_product = []
    for item in product_embeddings:
        uniqueness_product.append(1 / np.exp(kde.score_samples([item])[0]))

    return np.array(uniqueness_product)


def group_diversity(product_embeddings: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """Calculate group diversity based on kde uniqueness.

    Parameters
    ----------
    item_embedding_space: np.ndarray :
        item embeddings for estimate uniqueness

    Returns
    -------
    Tuple[bool, float]
        reject
        group diverstity

    """
    mean_diversity = np.sum(kde_uniqueness(product_embeddings)) / len(product_embeddings)
    reject = mean_diversity >= threshold
    return bool(reject), mean_diversity


def main() -> None:
    """Run application"""
    uvicorn.run(app, host="localhost", port=5000)


if __name__ == "__main__":
    main()
