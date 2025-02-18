import time

def topo_sort(node):
    visited = set()
    sorted_nodes = []
    
    def dfs(node):
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                if child is not None:
                    dfs(child)
            sorted_nodes.append(node)
                    
    dfs(node)

    return sorted_nodes


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        stop_time = time.time()
        
        print(f"[PERF] {func.__name__.upper()} took {stop_time - start_time:.4f} seconds")
        
        return result
    
    return wrapper        
    
    