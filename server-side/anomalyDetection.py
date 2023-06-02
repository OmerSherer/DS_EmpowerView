from __future__ import division

import itertools
from functools import reduce

import numpy as np
from cv2 import BORDER_CONSTANT, arrowedLine, circle, copyMakeBorder
from pandas import read_csv
from scipy.stats import median_abs_deviation
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import cuda, device, from_numpy, no_grad
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


def _intervals_extract(iterable):
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        yield [group[0][1], group[-1][1]]


def _ZRscore_outlier(df):
    out = []
    df = df.T
    for col in df:
        med = np.median(col)
        ma = median_abs_deviation(col)
        for index, i in enumerate(col):
            z = (0.6745 * (i - med)) / (np.median(ma))
            if np.abs(z) > 3:
                out.append(index)
    return sorted(list(set(out)))


class _RNN(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


class _TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def _movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def analyzeVideo(csvPath):
    """
    Analyze anomlies in certain video given its csv embeddings.

    ### Args:

        csvPath (`string`): Path to csv file which contains the embeddings.

    ### Returns:

    - `int`: number of outliers in the video.
    - `list[list[int]]`: intervals of anomalies in the video (example: [[71, 71], [166, 168]]).
    - `np.array[float]`: array which points the time for Y (X).
    - `np.array[float]`: array which points the chances of anomlies over time (Y).
    - y_predictions
    - expectedPoints
    - actualPoints

    (Last three for `writeOnFrameAnomalies` function).
    """

    framesPerSecond = 30
    df = read_csv(csvPath)

    try:
        df_noTimestamp = df.drop(["timestamp"], axis=1)
    except KeyError:
        df_noTimestamp = df

    df_noZero = df_noTimestamp[df_noTimestamp.columns[~(df_noTimestamp.sum() == 0)]]

    sc = StandardScaler()
    df_scaled = sc.fit_transform(df_noZero)

    timestamps = df_scaled.shape[0]

    num_of_cols = df_scaled.shape[1]

    pca = PCA(n_components=0.999)
    pca.fit(df_scaled)

    X = pca.transform(df_scaled)

    ZRScoreOutliers = _ZRscore_outlier(X)
    isZRScoreOutliers = []
    for i in range(timestamps):
        if i in ZRScoreOutliers:
            isZRScoreOutliers.append(1)
        else:
            isZRScoreOutliers.append(0)
    isZRScoreOutliers = np.array(isZRScoreOutliers, dtype=np.bool8)

    outliers_fraction = 0.01

    anomaly_algorithms = [
        (
            "Robust covariance",
            EllipticEnvelope(contamination=outliers_fraction, random_state=42),
        ),
        (
            "One-Class SVM",
            OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1),
        ),
        (
            "One-Class SVM (SGD)",
            make_pipeline(
                Nystroem(gamma=0.1, random_state=42, n_components=150),
                SGDOneClassSVM(
                    nu=outliers_fraction,
                    shuffle=True,
                    fit_intercept=True,
                    random_state=42,
                    tol=1e-6,
                ),
            ),
        ),
        (
            "Isolation Forest",
            IsolationForest(contamination=outliers_fraction, random_state=42),
        ),
        (
            "Local Outlier Factor",
            LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction),
        ),
    ]

    y_preds = []

    for name, algorithm in anomaly_algorithms:
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)
        y_preds.append(y_pred)

    input_size = num_of_cols
    hidden_size = 64
    output_size = num_of_cols
    sequence_length = 60
    learning_rate = 0.001
    num_epochs = 10
    used_device = device("cuda:0" if cuda.is_available() else "cpu")

    model = _RNN(input_size, hidden_size, output_size).to(used_device)

    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    X = []
    Y = []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i : i + sequence_length])
        Y.append(df_scaled[i + sequence_length])
    X = np.array(X)
    Y = np.array(Y)
    X = from_numpy(X).float()
    Y = from_numpy(Y).float()

    dataset = _TimeSeriesDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    running_loss = 0.0
    for epoch in range(num_epochs):
        for i in dataloader:
            inputs, outputs = i
            inputs = inputs.to(used_device)
            outputs = outputs.to(used_device)

            predictions = model(inputs)

            loss = criterion(predictions, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    running_loss = 0.0

    with no_grad():
        predictions = model(X.to(used_device))
    predictions = predictions.cpu().numpy()

    absoluteErrors = np.abs(predictions - Y.numpy())
    meanErrors = np.mean(absoluteErrors, axis=1).argsort()
    topOnePrecent = int((timestamps - sequence_length) * (1 - outliers_fraction))
    rnnAnomalies = meanErrors[topOnePrecent:] + sequence_length

    RNNColumnAnomalies = (
        absoluteErrors >= np.quantile(absoluteErrors, 0.99, axis=1)[:, None]
    )

    indexesRNNColumnAnomalies = []
    for i in RNNColumnAnomalies:
        indexesRNNColumnAnomalies.append(np.arange(RNNColumnAnomalies.shape[1])[i])

    XYIndexAnomalies = []
    for indexesOfAnomalies in indexesRNNColumnAnomalies:
        currentAnomalies = []
        for index in indexesOfAnomalies:
            temp = index
            while df_noZero.columns[temp][0] != "x":
                temp -= 1
            currentAnomalies.append([temp, temp + 1])
        XYIndexAnomalies.append(currentAnomalies)

    expectedPoints = []
    unscaledPredictions = sc.inverse_transform(predictions)
    for sliced, indexes in zip(unscaledPredictions, XYIndexAnomalies):
        points = []
        for indexPoint in indexes:
            try:
                points.append(sliced[indexPoint])
            except IndexError as e:
                pass
        expectedPoints.append(points)

    actualPoints = []
    for sliced, indexes in zip(df_noZero.values[sequence_length:], XYIndexAnomalies):
        points = []
        for indexPoint in indexes:
            try:
                points.append(sliced[indexPoint])
            except IndexError as e:
                pass
        actualPoints.append(points)

    expectedPoints = [[]] * sequence_length + expectedPoints
    actualPoints = [[]] * sequence_length + actualPoints

    isRNNOutliers = []
    for i in range(timestamps):
        if i in rnnAnomalies:
            isRNNOutliers.append(0)
        else:
            isRNNOutliers.append(1)
    isRNNOutliers = np.array(isRNNOutliers, dtype=np.bool8)

    y_preds.append(isZRScoreOutliers * 2 - 1)
    y_preds.append(isRNNOutliers * 2 - 1)
    y_preds.append(isRNNOutliers * 2 - 1)
    y_preds.append(isRNNOutliers * 2 - 1)

    outliers = reduce(lambda num1, num2: num1 + num2, y_preds)

    y_predictions = outliers < 0

    y = outliers * -1
    y = y - y.min()
    y = y / y.max()
    x = np.arange(len(y)) / framesPerSecond

    y_av = _movingaverage(y, framesPerSecond * 2)

    return (
        y_predictions.sum(),
        list(_intervals_extract(list(np.where(y_predictions)[0]))),
        x[sequence_length:-sequence_length],
        y_av[sequence_length:-sequence_length] * 100,
        y_predictions,
        expectedPoints,
        actualPoints,
    )


def writeAnomaliesOnFrame(
    cap,
    frame,
    index: int,
    border_size: int,
    y_predictions,
    expectedPoints,
    actualPoints,
):
    """
    Draw on a frame certain anomlies (if exists). Should be called on each image.

    ### Args:

        - cap (`VideoCapture`): The video capture input.
        - frame (`image`): The frame that that will be drawn on (in an immutable way).
        - index (`int`): current index of video.
        - border_size (`int`): The size for the drawn border.
        - y_predictions.
        - expectedPoints.
        - actualPoints.

    (Last three from `analyzeVideo` function.)

    ### Returns:

    - `image`: final frame.
    - `boolean`: is the current image was an anomaly.
    """

    if y_predictions[index]:
        for expected, actual in zip(expectedPoints[index], actualPoints[index]):
            frame = arrowedLine(
                frame,
                (
                    round(actual[0] * int(cap.get(3))),
                    round(actual[1] * int(cap.get(4))),
                ),
                (
                    round(expected[0] * int(cap.get(3))),
                    round(expected[1] * int(cap.get(4))),
                ),
                (255, 0, 0),
                1,
            )
            frame = circle(
                frame,
                (
                    round(actual[0] * int(cap.get(3))),
                    round(actual[1] * int(cap.get(4))),
                ),
                radius=1,
                color=(0, 255, 255),
                thickness=-1,
            )
            frame = circle(
                frame,
                (
                    round(expected[0] * int(cap.get(3))),
                    round(expected[1] * int(cap.get(4))),
                ),
                radius=1,
                color=(34, 255, 0),
                thickness=-1,
            )

        frame = copyMakeBorder(
            frame,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=BORDER_CONSTANT,
            value=[0, 0, 255],
        )
    else:
        frame = copyMakeBorder(
            frame,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=BORDER_CONSTANT,
            value=[0, 0, 0],
        )

    return frame, y_predictions[index]
