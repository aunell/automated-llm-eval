from scipy import stats
import pandas as pd
import math

def fleiss_kappa(mat):
    """ Computes the Kappa value
        @param n Number of rating per subjects (number of human raters)
        @param mat Matrix[subjects][categories]
        @return The Kappa value """

    def checkEachLineCount(mat):
        """ Assert that each line has a constant number of ratings
            @param mat The matrix checked
            @return The number of ratings
            @throws AssertionError If lines contain different number of ratings """
        n = sum(mat[0])
        
        assert all(sum(line) == n for line in mat[1:]), "Line count != %d (n value)." % n
        return n

    n = checkEachLineCount(mat)   # PRE : every line count must be equal to n
    N = len(mat)
    k = len(mat[0])
    
    # Computing p[j]
    p = [0.0] * k
    for j in range(k):
        p[j] = 0.0
        for i in range(N):
            p[j] += mat[i][j]
        p[j] /= N*n
    
    # Computing P[]    
    P = [0.0] * N
    for i in range(N):
        P[i] = 0.0
        for j in range(k):
            P[i] += mat[i][j] * mat[i][j]
        P[i] = (P[i] - n) / (n * (n - 1))
    # if DEBUG: print "P =", P
    
    # Computing Pbar
    Pbar = sum(P) / N
    # if DEBUG: print "Pbar =", Pbar
    
    # Computing PbarE
    PbarE = 0.0
    for pj in p:
        PbarE += pj * pj
    try:
        kappa = (Pbar - PbarE) / (1 - PbarE)
    except:
        kappa=1
    return kappa

def add_fleiss_column(file):
     df = pd.read_csv(file)
     index_to_scores = calculate_fleiss_kappa(file)
     df['Scores'] = df['idx'].map(index_to_scores)
     df.to_csv("VanDeenCollapsed_updated.csv", index=False)

def calculate_fleiss_kappa(dataset):
    """
    Calculate Fleiss' Kappa for each text corpus.

    Parameters:
    - dataframe: Pandas DataFrame containing the data.
    - column_name: Name of the column containing the ratings.

    Returns:
    - Dictionary mapping text corpus index to Fleiss' Kappa value.
    """

    def aggregate_for_fleiss(list_scores):
        return_table = []
        for score in list_scores:
            if score == -2:
                temp=[1,0,0,0,0]
            elif score ==-1:
                temp=[0,1,0,0,0]
            elif score == 0:
                temp=[0,0,1,0,0]
            elif score == 1:
                temp=[0,0,0,1,0]
            else:
                temp=[0,0,0,0,1]
            return_table.append(temp)
        return [[sum(x) for x in zip(*return_table)]]

    # Pivot the DataFrame to get a matrix of ratings
    df = pd.read_csv(dataset)
    pivot_df = df.pivot(index='reader', columns='idx', values='q2')
    # Calculate Fleiss' Kappa for each text corpus
    kappa_values = {}
    for idx, column_data in pivot_df.items():
        # Extract ratings for the current text corpus
        ratingsList = list(column_data.values)
        filtered_list = [value for value in ratingsList if not math.isnan(value)]
        agg = aggregate_for_fleiss(filtered_list)
        fleiss_kappa_res = fleiss_kappa(agg)
        kappa_values[idx] = fleiss_kappa_res

    return kappa_values