import numpy as np

roll_no = 1024170069
print("Roll No:", roll_no)

print("\nQ1")
a1 = np.array([4, 8, 12, 16, 20])
print("Original:", a1)
print("Add 2:", a1 + 2)
print("Multiply by 3:", a1 * 3)
print("Divide by 2:", a1 / 2)

print("\nQ2")
arr = np.array([1, 2, 3, 6, 4, 5])
print("Reversed:", arr[::-1])


def most_frequent(a):
    values, counts = np.unique(a, return_counts=True)
    top = counts.max()
    for v in values[counts == top]:
        print("Most frequent value:", v, "| count:", top, "| indices:", np.where(a == v)[0])


x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])
print("x:")
most_frequent(x)
print("y:")
most_frequent(y)

print("\nQ3")
arr2d = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print("1st row, 2nd column:", arr2d[0, 1])
print("3rd row, 1st column:", arr2d[2, 0])

print("\nQ4")
daksh_chhabra = np.linspace(10, 100, 25)
print("Array:", daksh_chhabra)
print("Dimensions:", daksh_chhabra.ndim)
print("Shape:", daksh_chhabra.shape)
print("Total elements:", daksh_chhabra.size)
print("Data type:", daksh_chhabra.dtype)
print("Total bytes:", daksh_chhabra.nbytes)
transposed = daksh_chhabra.reshape(25, 1)
print("Transpose using reshape, shape:", transposed.shape)
print(transposed)
print("Shape using T attribute:", daksh_chhabra.T.shape)
print("T gives the same 1-D array:", np.array_equal(daksh_chhabra.T, daksh_chhabra))
print("T cannot turn a 1-D array into a column; use reshape(25, 1) or daksh_chhabra[:, np.newaxis].")

print("\nQ5")
ucs420_daksh_chhabra = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 15, 20, 35]).reshape(3, 4)
print("Array:\n", ucs420_daksh_chhabra)
print("Mean:", np.mean(ucs420_daksh_chhabra))
print("Median:", np.median(ucs420_daksh_chhabra))
print("Max:", np.max(ucs420_daksh_chhabra))
print("Min:", np.min(ucs420_daksh_chhabra))
print("Unique elements:", np.unique(ucs420_daksh_chhabra))
reshaped_ucs420_daksh_chhabra = ucs420_daksh_chhabra.reshape(4, 3)
print("Reshaped (4x3):\n", reshaped_ucs420_daksh_chhabra)
resized_ucs420_daksh_chhabra = np.resize(ucs420_daksh_chhabra, (2, 3))
print("Resized (2x3):\n", resized_ucs420_daksh_chhabra)
