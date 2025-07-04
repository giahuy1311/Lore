import networkx as nx
import torch

def levenshtein_ed(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    
                                   dp[i][j-1],    
                                   dp[i-1][j-1])  

    return dp[m][n]

def graph_to_string(G):
    atom_types = ["C", "N", "O", "F", "I", "Cl", "Br"]
    node_parts = []
    for n in sorted(G.nodes()):
        x = G.nodes[n].get("x")
        if x is not None:
            x_tensor = torch.tensor(x)
            atom_idx = x_tensor.argmax().item()
            label = atom_types[atom_idx]
        else:
            label = "None"
        node_parts.append(label)
    
    edge_parts = []
    max_node = max(G.nodes())
    for i in range(max_node+1):
        for j in range(i, max_node+1):
            if G.has_edge(i, j):
                edge_parts.append("1")
            else:
                edge_parts.append("0")
            

    return "".join(node_parts) + "".join(edge_parts)

def visualize_mutag_graph(G, pos=None, title=None, ax=None, save_path=None,
                          important_edges=None, important_nodes=None, node_label = False):
    import matplotlib.pyplot as plt
    import torch

    atom_colors = {
        'C': "#FCA709",
        'N': '#3776AB',
        'O': '#FF6961',
        'F': '#ADD8E6',
        'I': '#8A2BE2',
        'Cl': '#228B22',
        'Br': '#FF6347'
    }
    atom_types = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
    node_labels = {}
    node_colors = []
    node_border_colors = []
    node_border_widths = []

    important_nodes = set(important_nodes) if important_nodes else set()
    important_edges = set(important_edges) if important_edges else set()

    for node in G.nodes():
        node_feature = G.nodes[node].get('x')
        node_feature = torch.tensor(node_feature)
        if node_feature is not None:
            atom_idx = node_feature.argmax().item()
            atom = atom_types[atom_idx]
        else:
            atom = 'C'
        node_labels[node] = f"{atom}"
        node_colors.append(atom_colors.get(atom, '#808080'))

        if f"nL_{node}" in important_nodes:
            node_border_colors.append("black")
            node_border_widths.append(2.5)
        else:
            node_border_colors.append("gray")
            node_border_widths.append(0.5)

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        edgecolors=node_border_colors,
        linewidths=node_border_widths,
        node_size=300,
        ax=ax
    )

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        edge = tuple(sorted((u, v)))
        if f"n{edge[0]}_n{edge[1]}" in important_edges:
            edge_colors.append("black")
            edge_widths.append(5)
        else:
            edge_colors.append("gray")
            edge_widths.append(1)

    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color=edge_colors,
        ax=ax
    )

    if node_label:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, ax=ax)
    else:
        nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)

    if title:
        ax.set_title(title, fontsize=10, color='black')

    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()