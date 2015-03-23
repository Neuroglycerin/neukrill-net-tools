# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:19:31 2015

@author: matt
"""

import ete2
from taxonomy import taxonomy

def build_tree_from_dict(dict_tree, tree=None):
    if tree is None:
        tree = ete2.Tree(name="root")
    for parent, children in dict_tree.iteritems():
        subtree = tree.add_child(name=parent)
        if children:
            subtree = build_tree_from_dict(children, subtree)
    return tree
    
def named_internal_node_layout(node):
    if node.is_leaf():
         name_face = ete2.AttrFace("name")
    else:
         name_face = ete2.AttrFace("name", fsize=10)
    ete2.faces.add_face_to_node(name_face, node, column=0, position="branch-right")
    
def make_equal_depth_tree(tree):
    tree = tree.copy()
    leaves = tree.get_leaves()
    depths = [tree.get_distance(leaf) for leaf in leaves]
    equalised = max(depths) == min(depths)
    while not equalised:
        min_depth = min(depths)
        for i, leaf in enumerate(leaves):
            if depths[i] == min_depth:
                leaves[i] = leaf.add_child(name=leaf.name)
                leaves[i].img_style['fgcolor'] = 'red'
                depths[i] += 1
        equalised = max(depths) == min(depths)
    return tree
    
def make_equal_depth_tree_bottom_up(tree):
    tree = tree.copy()
    nodes = tree.get_leaves()
    parents = set([node.up for node in nodes])
    while parents != set([tree.get_tree_root()]):
        for parent in parents:
            children = set(parent.get_children())
            children_in_nodes =  children.intersection(set(nodes))
            if children_in_nodes != children:
                map(lambda n: n.detach(), children_in_nodes)
                name = '\n'.join([n.name for n in children_in_nodes])
                new_parent = ete2.TreeNode(name=name)
                new_parent.img_style['fgcolor'] = 'red'
                map(lambda n: new_parent.add_child(n), children_in_nodes)
                parent.add_child(new_parent)
                parents.remove(parent)
                parents.add(new_parent)
        nodes = parents
        parents = set([parent.up for parent in parents])
    return tree
    

def get_equal_depth_tree_layers(tree):
    leaves = tree.get_leaves()
    depths = [tree.get_distance(leaf) for leaf in leaves]
    if min(depths) != max(depths):
        raise Exception('Tree with equal depth leaf nodes required')
    layers = [ set(leaves) ]
    while next(iter(layers[-1])).up is not tree:
        layers.append(list(set([node.up for node in layers[-1]])))
    layers = layers[::-1]
    for l, layer in enumerate(layers):
        for k, node in enumerate(layer):
            node.layer = l
            node.layer_index = k
    return layers
    
def change_tree_node_size(tree, size):
    for node in tree.traverse():
        node.img_style['size'] = size
        
ts = ete2.TreeStyle()
ts.mode = 'c'
#ts.show_leaf_name = False
#ts.layout_fn = named_internal_node_layout
ts.scale = None
ts.optimal_scale_level = 'full'
tree = build_tree_from_dict(taxonomy)
eq_tree_td = make_equal_depth_tree(tree)
change_tree_node_size(eq_tree_td, 10)
eq_tree_bu = make_equal_depth_tree_bottom_up(tree)
change_tree_node_size(eq_tree_bu, 10)
#layers = get_equal_depth_tree_layers(eq_tree)
eq_tree_td.show(tree_style=ts)
eq_tree_bu.show(tree_style=ts)