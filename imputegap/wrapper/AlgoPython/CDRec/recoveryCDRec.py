#!/usr/bin/python3

import numpy as np

# Centroid Recovery - uses truncated CD to impute all missing values in the matrix (designated with NaN)
def centroid_recovery(matrix, truncation = 0, maxIterations = 100, threshold = 1E-6):
    # input processing
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    n = len(matrix)
    m = len(matrix[0])
    
    if truncation > m: #strictly bigger
        print("[Centroid Recovery] Error: provided truncation parameter k=" + str(truncation) + " is larger than the number of columns m=" + str(m))
        print("[Centroid Recovery] Aborting recovery. Please provide a valid truncation parameter 1 <= k <= m - 1.")
        print("[Centroid Recovery] Alternatively, providing k = 0 or k = m will choose one automatically.")
        return None
    
    if truncation == 0 or truncation == m:
        truncation = 3
    
    truncation = min(truncation, m - 1)
    truncation = max(truncation, 1)
    maxIterations = max(maxIterations, 1)
    
    miss_mask = np.isnan(matrix)
    miss_count = sum(sum(miss_mask))
    
    if miss_count == 0:
        print("[Centroid Recovery] Warning: provided matrix doesn't contain any missing values.")
        print("[Centroid Recovery] The algorithm will run, but will return an unchanged matrix.")
    
    # initiate missing values
    matrix = interpolate(matrix, miss_mask)
    
    # init persistent values
    SV = default_SV(n, truncation)
    iter = 0
    last_diff = threshold + 1.0 #dummy to ensure it doesn't terminate in 1 hop
    #last_diff = 99.0 > small change - not update (NQU)
    
    # main loop
    while iter < maxIterations and last_diff >= threshold:
        # terminated if we reach the interation cap
        # or if our change to missing values from last iteration is small enough
        iter += 1
        
        # perform truncated decomposition
        res = centroid_decomposition(matrix, truncation, SV)
        
        if res == None: #make sure it doesn't fail, if it does - fail as well
            return None
        else:
            (L, R, SV) = res
        
        # perform a low-rank reconstruction of the original matrix
        recon = np.dot(L, R.T)
        
        # compute how much did it change using ||X[mis] - Xrec[mis]||_F / sqrt(|mis|)
        diff_vector = matrix[miss_mask] - recon[miss_mask]

        # last_diff = np.linalg.norm(diff_vector) / np.sqrt(miss_count) -> # big change > updated (NQU)
        last_diff = np.mean(np.abs(diff_vector))
        
        # substitute values in the missing blocks with what was reconstructed after truncated CD
        matrix[miss_mask] = recon[miss_mask]
    #end while
    
    return matrix
#end function


# simple linear interpolation function
# interpolates segments which are marked as NaN
# if the segments start (or ends) at the start (or end) of the column - uses 1NN instead
def interpolate(matrix, mask):
    n = len(matrix)
    m = len(matrix[0])
    
    for j in range(0, m):
        mb_start = -1
        prev_value = np.nan
        step = 0 #init
        
        for i in range(0, n):
            if mask[i][j]:
                # current value is missing - we either start a new block, or we are in the middle of one

                if mb_start == -1:
                    # new missing block
                    mb_start = i
                    mb_end = mb_start + 1
                    
                    while (mb_end < n) and np.isnan(matrix[mb_end][j]):
                        mb_end += 1
                    
                    next_value = np.nan if mb_end == n else matrix[mb_end][j]

                    if mb_start == 0: # special case #1: block starts with array
                        prev_value = next_value
                    
                    if mb_end == n: # special case #2: block ends with array
                        next_value = prev_value

                    step = (next_value - prev_value) / (mb_end - mb_start + 1) # small change > not updated (NQU)
                #end if

                matrix[i][j] = prev_value + step * (i - mb_start + 1)
            else:
                # missing block either ended just now or we're traversing normal data
                prev_value = matrix[i][j]
                mb_start = -1
            #end if
        #end for
    #end for
    
    return matrix
#end function


##
## decomposition functions
##

# Centroid Decomposition, with the optional possibility of specifying truncation or usage of initial sign vectors
def centroid_decomposition(matrix, truncation = 0, SV = None):
    # input processing
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    n = len(matrix)
    m = len(matrix[0])
    
    if truncation == 0:
        truncation = m
    
    if truncation < 1 or truncation > m:
        print("[Centroid Decomposition] Error: invalid truncation parameter k=" + str(truncation))
        print("[Centroid Decomposition] Aboritng decomposition")
        return None
        
    if SV is None:
        SV = default_SV(n, truncation)
    
    if len(SV) != truncation:
        print("[Centroid Decomposition] Error: provided list of Sign Vectors doesn't match in size with the truncation truncation parameter k=" + str(truncation))
        print("[Centroid Decomposition] Aboritng decomposition")
        return None
    
    L = np.zeros((truncation, n))
    R = np.zeros((truncation, m))
    
    # main loop - goes up till the truncation param (maximum of which is the # of columns)
    for j in range(0, truncation):
        # calculate the sign vector
        Z = local_sign_vector(matrix, SV[j])
        
        # calculate the column of R by X^T * Z / ||X^T * Z||
        R_i = matrix.T @ Z
        R_i = R_i / np.linalg.norm(R_i)
        R[j] = R_i
        
        # calculate the column of L by X * R_i
        L_i = matrix @ R_i
        L[j] = L_i
        
        # subtract the dimension generated by L_i and R_i from the original matrix
        matrix = matrix - np.outer(L_i, R_i)
        
        # update the new sign vector in the array
        SV[j] = Z
    #end for


    return (L.T, R.T, SV)
#end function


# Algorithm: LSV (Local Sign Vector). Finds locally optimal sign vector Z, i.e.:
#   Z being locally optimal means: for all Z' sign vectors s.t. Z' is one sign flip away from Z at some index j,
#   we have that ||X^T * Z|| >= ||X^T * Z'||
def local_sign_vector(matrix, Z):
    n = len(matrix)
    m = len(matrix[0])
    eps = np.finfo(np.float64).eps
    
    Z = local_sign_vector_init(matrix, Z)
        
    # calculate initial product of X^T * Z with the current version of Z
    direction = matrix.T @ Z
    # calculate initial value of ||X^T * Z||
    lastNorm = np.linalg.norm(direction) ** 2 + eps
    
    flipped = True
    
    while flipped:
        # we terminate the loop if during the last pass we didn't flip a single sign
        flipped = False
        
        for i in range(0, n):
            signDouble = Z[i] * 2
            gradFlip = 0.0
            
            # calculate how ||X^T * Z|| would change if we would change the sign at position i
            # change to the values of D = X^T * Z is calculated as D_j_new = D_j - 2 * Z_i * M_ij for all j
            for j in range(0, m):
                localMod = direction[j] - signDouble * matrix[i][j]
                gradFlip += localMod * localMod
            
            # if it results in augmenting ||X^T * Z||
            # flip the sign and replace cached version of X^T * Z and its norm
            if gradFlip > lastNorm:
                flipped = True
                Z[i] = Z[i] * -1
                lastNorm = gradFlip + eps
                
                for j in range(0, m):
                    direction[j] -= signDouble * matrix[i][j]
                #end for
            #end if
        #end for
    #end while
    
    return Z
#end function


# Auxiliary function for LSV:
#   Z is initialized sequentiually where at each step we see which sign would give a larger increase to ||X^T * Z||
def local_sign_vector_init(matrix, Z):
    n = len(matrix)
    m = len(matrix[0])
    direction = matrix[0].copy() # small change > updated (NQU) + .copy()
    
    for i in range(1, n):
        gradPlus = 0.0
        gradMinus = 0.0
        
        for j in range(0, m):
            localModPlus = direction[j] + matrix[i][j]
            gradPlus += localModPlus * localModPlus
            localModMinus = direction[j] - matrix[i][j]
            gradMinus += localModMinus * localModMinus
                
        if gradMinus > gradPlus:
            Z[i] = -1
        
        for j in range(0, m):
            direction[j] += Z[i] * matrix[i][j]
    
    return Z
#end function


#initialize sign vector array with default values
def default_SV(n, k):
    # default sign vector is (1, 1, ..., 1)^T
    baseZ = np.array([1.0] * n)
    SV = []
    
    for i in range(0, k):
        SV.append(baseZ.copy())
    
    return SV
#end function


def main():
    matrix = np.loadtxt("data_miss.txt")
    recovered = centroid_recovery(matrix, 1)
    reference = np.loadtxt("data_full.txt");
    
    print("Recovery error:")
    print(np.linalg.norm(recovered - reference))
    np.savetxt("data_recov.txt", recovered, fmt="%10.5f")

if __name__ == "__main__":
     main()

