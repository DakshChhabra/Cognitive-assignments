import numpy as np

roll_no = 1024170069
print("Roll No:", roll_no)

print("\nQ1")
temperature = np.array([25, 28, 31, 35, 38, 27, 33, 40])
adjusted = temperature + 2
print("a. Adjusted (+2 C):", adjusted)
fahrenheit = (9 / 5) * adjusted + 32
print("b. Fahrenheit:", fahrenheit)
hot = temperature[temperature > 32]
print("c. Readings > 32 C:", hot)
print("d. Count of readings > 32 C:", np.count_nonzero(temperature > 32))
print("e. Vectorization applies an operation to the whole array at once in compiled C code,")
print("   avoiding the per-element overhead of a Python loop. Boolean indexing builds a mask")
print("   and selects all matching values in one step, so filtering and counting need no")
print("   explicit iteration, giving shorter, clearer and much faster code on large data.")

print("\nQ2")
steps = np.array([
    [5000, 6200, 7100],
    [8000, 7500, 9000],
    [4500, 5100, 4800],
    [9000, 8500, 9500]
])
print("a. Total steps:", steps.sum())
print("b. Mean steps:", steps.mean())
print("c. Max:", steps.max(), "| Min:", steps.min())
print("d. Total per day (axis=0):", steps.sum(axis=0))
print("e. Total per user (axis=1):", steps.sum(axis=1))
pos = tuple(int(i) for i in np.unravel_index(steps.argmax(), steps.shape))
print("f. argmax (flat index):", steps.argmax(), "| (user, day) position:", pos, "| value:", steps[pos])

print("\nQ3")
original = np.array([1, 2, 3, 4, 5, 6])
subset = original[1:5]
print("b. subset:", subset)
subset[0] = 999
print("c. original:", original)
print("   subset:", subset)
original = np.array([1, 2, 3, 4, 5, 6])
copied = original[1:5].copy()
copied[0] = 500
print("d. original:", original)
print("   copied:", copied)
matrix = np.arange(1, 13).reshape(3, 4)
print("e. Matrix:\n", matrix)
print("f. First row:", matrix[0])
print("   Last row:", matrix[-1])
print("   Second column:", matrix[:, 1])
print("   Rows 1-2, columns 2-3:\n", matrix[0:2, 1:3])
flat = matrix.flatten()
rav = matrix.ravel()
print("g. flatten():", flat)
print("   ravel():", rav)
rav[0] = 100
print("h. After modifying ravel() result, matrix:\n", matrix)
print("   Original affected (ravel returns a view):", matrix[0, 0] == 100)
flat[1] = 200
print("i. After modifying flatten() result, matrix:\n", matrix)
print("   Original affected (flatten returns a copy):", matrix[0, 1] == 200)
print("j. shape:", matrix.shape, "| ndim:", matrix.ndim, "| size:", matrix.size, "| dtype:", matrix.dtype)

print("\nQ4")
X = np.array([
    [6, 70, 3],
    [5, 50, 6],
    [8, 80, 2],
    [4, 30, 8]
])
y = np.array([40, 65, 30, 85])
print("a. X shape:", X.shape, "| ndim:", X.ndim)
print("b. X.T:\n", X.T)
print("   The transpose swaps rows and columns: each row of X.T holds one feature")
print("   (sleep, activity, stress) across all four users.")
XtX = X.T @ X
print("c. X.T @ X:\n", XtX)
print("d. Determinant of X.T @ X:", np.linalg.det(XtX))
XtX_inv = np.linalg.inv(XtX)
print("   Inverse of X.T @ X:\n", XtX_inv)
beta = XtX_inv @ X.T @ y
print("e. beta = (X.T X)^-1 X.T y:", beta)
print("   Check with np.linalg.lstsq:", np.linalg.lstsq(X, y, rcond=None)[0])
print("f. Each coefficient is the change in predicted assistance score for a one-unit increase")
print("   in that feature with the others held fixed:")
print("   Sleep hours coefficient   :", round(beta[0], 4))
print("   Activity level coefficient:", round(beta[1], 4))
print("   Stress level coefficient  :", round(beta[2], 4))
print("   A positive coefficient raises the predicted score, a negative one lowers it.")
print("   The model has no intercept term.")
new_user = np.array([5, 40, 7])
prediction = new_user @ beta
print("g. Predicted assistance score for new user:", prediction)
