import math
# Iplementation of CART - Classification and Regression Trees by Leo Breiman

# Recursive Binary Splitting - Greedy Approach to split the space
# Regression - criteria for best split is based off Residual Sum of Squares. Pick a split, then the predictions for each of the new regions will be the mean
#   of that region. Calculate RSS. Find the split which minimizes RSS the most. 
# Classification - criteria for best split can be based off Gini Index or Entropy. These are both measures of node purity / entropy. 

# Gini Index implementation
def gini_index(groups, classes):
    #Count the number of total observations
    numInstances = float(sum([len(group) for group in groups]))

    if numInstances == 0:
        return 0.0

    gini = 0.0
    # Each group can be considered a region that is subdivided by the decision tree or as terminal node / leaf in CART
    for group in groups:
        size = float(len(group))
        proportion = size / numInstances
        if proportion == 0 :
            continue
        score = 0.0

        # Gini Index is calculated by summing for all classes
        #    proportion of observations of that class / total obs
        #    in that region
        for classVal in classes:
            specProp = [row[-1] for row in group].count(classVal) / size
            score += specProp * (1 - specProp)
        # weighing the score by the number of observations in that region vs num total obs
        gini += score * proportion
    return gini

# Shannon Entropy Implementation
def entropyCalc(groups, classes):
    numInstaces = float(sum( [len(group) for group in groups] ))

    if numInstaces == 0:
        return 0.0
    
    entropy = 0.0
    # For each region/terminal node in the decision tree.....
    for group in groups:
        size = float(len(group))
        proportion = size / numInstances

        if proportion == 0:
            continue
        
        score = 0.0

        for classVal in classes:
            specProp = [row[-1] for row in group].count(classVal) / size
            score += (-1.0) * specProp*( math.log(specProp) / math.log(2) ) # we use the proportions as a proxy for probabilities
        entropy += score
    return entropy

