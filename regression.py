import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def multiply_dataframes(data_frame1, data_frame2) -> pd.DataFrame:
    mat1 = data_frame1.values
    mat2 = data_frame2.values
    result = np.matmul(mat1, mat2)

    return pd.DataFrame(result)


def least_square(data_frame: pd.DataFrame):
    x = data_frame['x']
    y = data_frame['y']
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x * y)
    sum_x_squared = sum(x ** 2)
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - a * sum_x) / n
    return a, b


def iterate_least_squares(a, b, data_frame: pd.DataFrame) -> list:
    x = data_frame['x'].values
    y = data_frame['y'].values
    result = []
    for i in range(len(x)):
        read_value = y[i]
        estimated_value = a * x[i] + b
        error = abs(read_value - estimated_value)
        result.append([read_value, estimated_value, error])

    return result


if __name__ == '__main__':

    df = pd.read_csv('data.csv', header=None, names=['x', 'y'])

    # transpose the data frame
    train_size = int(0.95 * len(df))
    train_df = df.iloc[:train_size]
    result_df = df.iloc[train_size:]

    a, b = least_square(train_df)
    test_result = iterate_least_squares(a, b, result_df)

    for i in range(len(test_result)):
        print('Read value: {}'.format(test_result[i][0]),
              '\nEstimated value: {}'.format(test_result[i][0]),
              '\nError: {}'.format(test_result[i][0]))
        print()

    plt.plot(df['x'], df['y'], 'o')
    plt.plot(df['x'], a * df['x'] + b)
    plt.show()
