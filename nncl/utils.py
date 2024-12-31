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


        
    
    