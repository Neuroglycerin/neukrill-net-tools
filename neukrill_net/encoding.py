import taxonomy
import numpy as np

def get_hierarchy(settings=None):
    """
    Returns nested list of classes and superclasses,
    in string format. Required to use get_encoding.
    """

    # Big list with all arrays
    hierarchy = []

    # Array of classes at current depth
    classes = []

    # Array of classes at previous depth
    classes_prev = None

    # Set depth
    i = 0

    while classes != classes_prev:
        classes_prev = classes
        if classes:
            hierarchy.append(classes)
        layer = taxonomy.TaxonomyLayer(i)
        classes = []

        # Go through superclasses
        for key in sorted([cl for cl in layer]):
            # Add parent to classes if it's not already there
            if key not in classes:
                classes.append(key)
        i += 1

    hierarchy[0] = [str(c) for c in sorted(list(taxonomy.TaxonomyLayer(0)))]

    return hierarchy

def make_class_dictionary(classes, hierarchy):
    """
    Takes a list of classes and a hierarchy and makes
    a dictionary that'll map to integers for building
    target matrices easily.
    """
    # init dict
    class_dictionary = {}
    # iterate over classes
    for c in classes:
        # for each class collapse the nested list supplied
        # by get encoding and then take out the indices of
        # all the ones in it
        class_dictionary[c] = np.where(np.array([element
                        for lst in get_encoding(c,hierarchy)
                        for element in lst])==1)[0]
    return class_dictionary

def get_encoding(class_name, hierarchy):

    # Big list with all encoding vectors
    all_encodings = []

    for hier in hierarchy:
        # Put zeros in vector
        encoding = [0] * len(hier)
        i = 0

        while True:
            # Find the ancestor of class_name in hier
            layer = taxonomy.TaxonomyLayer(i)
            if layer[class_name] in hier:
                encoding[hier.index(layer[class_name])] = 1
                break
            i += 1

        # Useful for debugging: save as a dictionary of class labels and 0/1
        #class_map = {s:i for s,i in zip(hier, encoding)}
        #all_encodings.append(class_map)

        all_encodings.append(encoding)

    return all_encodings
