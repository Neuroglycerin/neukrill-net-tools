import taxonomy

def get_hierarchy():

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

    return hierarchy

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