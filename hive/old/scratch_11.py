import math

from pynauty import *
import torch
import numpy as np
import igraph as ig


def dist_matrix(x,precision=3):
    x = np.asarray(x)
    dist = np.sqrt((x**2).sum(axis=1)[:, None] - 2 * x.dot(x.transpose()) + ((x**2).sum(axis=1)[None, :]))
    return np.round(dist,decimals=precision)

def generate_all_possible_edge_len_dict(n,precision=3):
    x = torch.arange(n).repeat(n)
    y = torch.arange(n).repeat_interleave(n)
    xy = torch.stack((x,y),dim=1).numpy()
    D = dist_matrix(xy,precision)
    all_possible_edge_lens = np.unique(D)
    val = np.arange(len(all_possible_edge_lens))
    edge_color_lookup = dict(zip(all_possible_edge_lens, val))
    return edge_color_lookup

edge_color_lookup = generate_all_possible_edge_len_dict(5)#.tolist()

def generate_edges_and_edge_colors(coordinates):
    n = len(coordinates)

    D = dist_matrix(coordinates)
    edges = torch.combinations(torch.arange(n), 2).numpy()
    edge_distances = [D[edge[0], edge[1]] for edge in edges]
    edge_colors = [edge_color_lookup[edge] for edge in edge_distances]
    edges_list = edges.tolist()

    return D, edges.tolist(), edge_colors

def convert_edge_colors_to_vertices(node_colors_base,edges_base,edge_colors):
    edges_base = torch.tensor(edges_base)
    node_colors_base = torch.tensor(node_colors_base)
    nodes = len(node_colors_base)
    #First we determine the amount of layers needed
    color_set = set(edge_colors)
    n_colors = len(color_set)
    n_layers = math.ceil(math.log2(n_colors+1))
    node_colors = node_colors_base.repeat(n_layers).view(n_layers,-1)
    node_colors = node_colors+(torch.arange(n_layers)*node_colors_base.max())[:,None]
    edges = []
    #we connect all the trivial vertical edges between the layers
    for i in range(1,n_layers):
        a = torch.arange(len(node_colors[:-i].view(-1)))
        if i == 1:
            edges = torch.stack((a,a+nodes*i),dim=1)
        else:
            new_edges = torch.stack((a,a+nodes*i),dim=1)
            edges = torch.cat((edges,new_edges))
    for edge_base,edge_color in zip(edges_base,torch.tensor(edge_colors)):
        bitmask = binary(edge_color,n_layers).to(torch.bool)
        for i, bit in enumerate(bitmask):
            if bit:
                edge = edge_base+i*nodes
                edges = torch.cat((edges,edge[None,:]))
    return node_colors, edges

def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def get_canon_label(coordinates, node_colors):
    """
    This routine expects the node colors to be sorted such that all colors of the same kind are contiguous.
    """
    n = len(node_colors)
    D, edges, edge_colors = generate_edges_and_edge_colors(coordinates)
    nodes_extended, edges_extended = convert_edge_colors_to_vertices(node_colors,edges,edge_colors)
    nodes = nodes_extended.view(-1).tolist()
    vertex_coloring = [set() for _ in range(nodes[-1])]
    for i, node in enumerate(nodes):
        if i == 0:
            idx = 0
            node_old = node
        elif node != node_old:
            idx += 1
            node_old = node
        vertex_coloring[idx].add(i)
    adjacency_dict = {}
    for i in range(len(nodes)):
        adjacency_dict[i] = []
    edges_extended = edges_extended.tolist()
    for edge in edges_extended:
        adjacency_dict[edge[0]].append(edge[1])


    g = Graph(number_of_vertices=len(nodes),vertex_coloring=vertex_coloring,adjacency_dict=adjacency_dict,directed=False)
    # print(autgrp(g))
    print(f"Canonical permutation {canon_label(g)[:n]}")
    permute = canon_label(g)[:n]

    # node_colors_c = [node_colors[p] for p in permute]
    # edges_c = [[permute[edge[0]],permute[edge[1]]] for edge in edges]
    # edge_distances_c = [D[edge[0], edge[1]] for edge in edges_c]
    # edge_colors_c = [edge_color_lookup[edge] for edge in edge_distances_c]
    #
    rep = f"ORG =  node_colors={node_colors},edges={edges},edge_colors={edge_colors}"
    print(rep)
    # rep = f"Canon =  node_colors={node_colors_c},edges={edges_c},edge_colors={edge_colors_c}"
    # print(rep)
    permute_graph(permute, coordinates, node_colors)

    return rep

def permute_graph(permute,coordinates,node_colors):
    node_colors_c = [node_colors[p] for p in permute]
    coordinates_c = [coordinates[p] for p in permute]
    D, edges_c, edge_colors_c = generate_edges_and_edge_colors(coordinates_c)
    rep = f"Canon =  node_colors={node_colors_c},edges={edges_c},edge_colors={edge_colors_c}"
    print(rep)


coordinates = [[0,0],[-1,0],[1,0],[-1,1]]
node_colors = [1,2,2,2]
rep = get_canon_label(coordinates, node_colors)


coordinates = [[0,0],[1,0],[-1,0],[-1,1]] #Just a simple permutation of nodes, so should be isomorph
node_colors = [1,2,2,2]
rep = get_canon_label(coordinates, node_colors)

coordinates = [[1,1],[0,1],[2,1],[0,2]] #Just a simple translation of all nodes, so should be isomorph
node_colors = [1,2,2,2]
rep = get_canon_label(coordinates, node_colors)

coordinates = [[0,0],[-1,0],[1,0],[1,1]] # This is a mirroring so should also be isomorph
node_colors = [1,2,2,2]
rep = get_canon_label(coordinates, node_colors)

coordinates = [[0,0],[-2,0],[2,0],[-2,2]] # All distances are now twice as long, so this should be a new graph
node_colors = [1,2,2,2]
rep = get_canon_label(coordinates, node_colors)

coordinates = [[0,0],[-1,0],[1,0],[-1,1]] # Different node colors, so different graph
node_colors = [1,2,3,3]
rep = get_canon_label(coordinates, node_colors)

print("done")