import taxonomy

def create_encoding(class_name):

    # Sort classes alphabetically
    classes = sorted([key for key, value in taxonomy.superclasses.items()])

    # Make a 1-of-k array for classes
    class_array = [0] * len(classes)
    class_array[classes.index(class_name)] = 1

    # Big list with all arrays
    all_arrays = []
    #class_map = {s:i for s,i in zip(classes, class_array)}
    #all_arrays.append(class_map)
    all_arrays.append(class_array)

    # Set depth
    i = 1

    # Create an instance of TaxonomyLayer
    layer = taxonomy.TaxonomyLayer(i)

    # This will have all classes in it at the end of iterations
    check = []

    # While alphabetically sorted classes are not equal to the checking array
    while classes != check:
        check = []

        # This will be the next superclass array
        sup_classes = []
        # Go through superclasses
        for key, value in sorted(taxonomy.superclasses.items(), key = lambda x:x[0]):

            # Add first parent to sup_classes if it's not there yet
            if layer[key] != key:
                if not layer[key] in sup_classes:
                    sup_classes.append(layer[key])
            else:
                if not taxonomy.superclasses[key][-1] in sup_classes:
                    sup_classes.append(taxonomy.superclasses[key][-1])

            # Add first parent to check anyway
            check.append(layer[key])
       
        # Make a 1-of-k array for superclasses
        superclass_array = [0] * len(sup_classes)
        if layer[class_name] in sup_classes:
            superclass_array[sup_classes.index(layer[class_name])] = 1
        else:
            superclass_array[sup_classes.index(taxonomy.superclasses[key][-1])] = 1

        #superclass_map = {s:i for s,i in zip(sup_classes, superclass_array)}

        # Append it to the output list
        #all_arrays.append(superclass_map)
        all_arrays.append(superclass_array)

        # Increase depth
        i = i + 1
        layer = taxonomy.TaxonomyLayer(i)

    
    if all([x==y for x,y in zip(all_arrays[-1], all_arrays[-2])]):
        all_arrays.pop()
    return all_arrays