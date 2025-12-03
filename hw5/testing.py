import timeit
import numpy as np
import matplotlib.pyplot as plt

from vectorized_db import LSHDatabase


def test_lsh_2d(num_points=5000, num_results=100, L=10, k=7, batch_size=64):
    dim = 2
    lsh_db = LSHDatabase(dim, L, k)
    
    np.random.seed(42)
    points = np.random.uniform(low=-10., high=10., size=(num_points, 2))
    
    for i in range(0, len(points), batch_size):
        cur_batch = points[i:min(len(points), i + batch_size)]
        lsh_db.add_batch_vectors(cur_batch)
    print('lsh database created')
    
    query_point = np.random.uniform(low=-10., high=10., size=(2))
    
    # LSH
    lsh_time = timeit.timeit(lambda: lsh_db.search(query_point, num_results, use_lsh=True, return_indices=True),
                             number=1000) * 1e-3
    lsh_results, lsh_indices = lsh_db.search(query_point, num_results, use_lsh=True, return_indices=True)
    
    # Полный перебор
    dummy_time = timeit.timeit(lambda: lsh_db.search(query_point, num_results, use_lsh=False, return_indices=True), 
                               number=1000) * 1e-3
    dummy_results, dummy_indices = lsh_db.search(query_point, num_results, use_lsh=False, return_indices=True)

    print(f"LSH поиск: {lsh_time:.6f} сек")
    print(f"Полный перебор: {dummy_time:.6f} сек")
    print(f"Ускорение: {dummy_time/lsh_time:.2f}x")
    
    lsh_set = set(lsh_indices)
    dummy_set = set(dummy_indices)
    intersection = lsh_set.intersection(dummy_set)
    precision = len(intersection) / len(dummy_set)
    print(f"LSH accuracy: {precision}")
    
    plt.figure(figsize=(12, 4))
    
    # Результаты LSH
    plt.subplot(1, 3, 1)
    plt.scatter(points[:, 0], points[:, 1], 
                s=5, alpha=0.7, linewidths=0,
                color='gray')
    plt.scatter(lsh_results[:, 0], lsh_results[:, 1], 
                s=3, alpha=0.5,
                color='blue', label='LSH результаты')
    plt.scatter(query_point[0], query_point[1], 
                color='red', s=75, marker='*', 
                label='Запрос')
    plt.title(f'LSH поиск ({lsh_time:.4f} сек)')
    plt.legend()
    
    # Результаты полного перебора
    plt.subplot(1, 3, 2)
    plt.scatter(points[:, 0], points[:, 1], 
                s=5, alpha=0.7, linewidths=0,
                color='gray')
    plt.scatter(dummy_results[:, 0], dummy_results[:, 1], 
                s=3, alpha=0.5,
                color='green', label='Полный перебор')
    plt.scatter(query_point[0], query_point[1], 
                color='red', s=75, marker='*', 
                label='Запрос')
    plt.title(f'Полный перебор ({dummy_time:.4f} сек)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()    


def test_search(database, embedder, test_queries, dataset, num_results=3, print_recipe=True):
    
    for i, query in enumerate(test_queries):
        print(f'Запрос {i+1}:')
        print(query)
        query_embedding = embedder.encode([query])[0]
        
        _, res_idxs = database.search(
            query=query_embedding,
            num_results=num_results,
            use_lsh=True,
            return_indices=True
        )
        
        if len(res_idxs) == 0:
            print("Результатов не найдено.")
        else:
            for j, res_idx in enumerate(res_idxs):
                recipe_id = database.metadata[res_idx]['id']
                print(f"{j+1}. Рецепт № {recipe_id}, название: {dataset['name'][recipe_id]}")
                if print_recipe:
                    print(f"  Рецепт: {dataset['text'][recipe_id]}")
                    print()
        print('\n\n')
