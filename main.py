'''
    Author: Krischin Layon

    practice_naiveBayes

    a practice naive bayes classifier from scratch for demonstration purposes

'''

import numpy as np

def quantitativeRange(vals,ranges):
    #vals = an n-sized list of quantitative values
    #ranges = a list of 3-tuples. the first item is the lower bound (inclusive), the second item is the upper bound (inclusive), and the last item is the string of the category. ex. "medium distance"
    #     -an 'i' means negative or positive infinity. (on the lower bound, it means negative. if the upper bound, it means positive)
    #returns an n-size list of categorical strings based on the quantitative values
    categories = []

    for v in vals:
        passFails = 0
        for r in ranges:
            inBoundPasses = 0 #get 2 and you pass

            #test lower bound
            if r[0] == 'i': inBoundPasses += 1
            elif v >= r[0]: inBoundPasses += 1

            #test upper bound
            if r[1] == 'i': inBoundPasses += 1
            elif v <= r[1]: inBoundPasses += 1

            #if the value in the appropriate range
            if inBoundPasses == 2:
                categories.append(r[2])
                break
            else: passFails += 1
            
        if passFails >= len(ranges):
            print("WARNING: \"" + str(v) + "\" does not fit in any of the provided ranges. Inputing as \"N/A\"")
            categories.append("N/A")

    return categories

def classify():
    #1st argument is a n-width feature matrix of categorical strings.
    #2nd argument is a response vector of strings for training. 
    #    -each column of the feature matrix is a different categorical variable. each row is an entry
    #3rd argument is a n-tuple of input features to attempt to classify
    
    #returns a response string based on the 3rd argument
    print("TODO")


def main():
    testValues = [1,10,100,10,1,1000]
    conv = [(0,9,"Ones"),(10,99,"Tens"),(100,999,"Hundreds")]
    print(quantitativeRange(testValues,conv))

main()