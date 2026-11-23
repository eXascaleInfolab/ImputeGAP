# ===============================================================================================================
# SOURCE: https://github.com/eXascaleInfolab/bench-vldb20
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# The code was transformed from the C++ original version to a python version
# ===============================================================================================================


"""Faithful NumPy translation of the supplied C++ ST_MVL implementation.

Input shape is (time steps, stations), exactly like the Armadillo source.
Only NaN is considered missing. The C++ model constructs synthetic geographic
coordinates from the column indices; this translation intentionally does too.

Normal numeric outputs should agree up to floating-point effects. C++ unsigned
integer overflow, non-finite map keys and undefined behavior are not emulated.
"""
from __future__ import annotations

import math
import numpy as np

EARTH_RADIUS = 6378137.0


def _div(a, b):
    """IEEE-style floating division for the divisions present in the C++ code."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return float(np.float64(a) / np.float64(b))


def _sqrt(x):
    with np.errstate(invalid='ignore'):
        return float(np.sqrt(np.float64(x)))


class ST_MVL:
    temporal_threshold = 5
    viewCount = 4

    def __init__(self, missing, alpha, gamma, windowSize):
        self.missingMatrix = np.asarray(missing, dtype=np.float64).copy()
        if self.missingMatrix.ndim != 2:
            raise ValueError('matrix must have two dimensions')
        self.rowCount, self.columnCount = self.missingMatrix.shape
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.windowSize = int(windowSize)
        if self.windowSize < 0:
            raise ValueError('windowSize must be nonnegative (C++ accepts size_t)')
        self.predictMatrix = self.missingMatrix.copy()
        self.temporaryMatrix = self.missingMatrix.copy()
        self.distanceDic = {}
        self.stationDic = {}

        latitude, longitude = [], []
        lat, lng = 39.954047, 116.348991
        for i in range(self.columnCount):
            latitude.append(lat)
            longitude.append(lng)
            if i % 2 == 0:
                lat += 0.01
            else:
                lng += 0.01

        for i in range(self.columnCount):
            candidates = []
            for j in range(self.columnCount):
                if i != j:
                    dis = self.GeoDistance(latitude[i], longitude[i], latitude[j], longitude[j])
                    self.distanceDic[(i, j)] = dis
                    candidates.append(j)
            # C++ intentionally doesn't sort by distance: insertion order.
            self.stationDic[i] = candidates

    @staticmethod
    def ismissing(val):
        return math.isnan(float(val))

    def ComputeSpatialWeight(self, dis):
        return float(np.power(_div(1.0, dis), self.alpha))

    def ComputeTemporalWeight(self, timespan):
        return self.gamma * (1.0 - self.gamma) ** (timespan - 1)

    @staticmethod
    def RadToDeg(d):
        # Name is misleading in the original, but formula is degrees -> radians.
        return d * math.pi / 180.0

    def GeoDistance(self, lat1, lng1, lat2, lng2):
        radLat1 = self.RadToDeg(lat1)
        radLat2 = self.RadToDeg(lat2)
        a = radLat1 - radLat2
        b = self.RadToDeg(lng1) - self.RadToDeg(lng2)
        inside = math.sin(a / 2.0) ** 2 + math.cos(radLat1) * math.cos(radLat2) * math.sin(b / 2.0) ** 2
        return 2.0 * math.asin(math.sqrt(inside)) * EARTH_RADIUS

    @staticmethod
    def _candidate_average(candidates):
        if not candidates:
            return math.nan
        # std::map<double,double> iterates in ascending order of its value keys.
        values = sorted(candidates.items())
        weightSum = 0.0
        for _, weight in values:
            weightSum += weight
        total = 0.0
        for value, weight in values:
            total += _div(value * weight, weightSum)
        return total

    def IDW(self, ti, tj, dataMatrix):
        candidates = {}
        for jj in self.stationDic[tj]:
            dis = self.distanceDic[(tj, jj)]
            value = float(dataMatrix[ti, jj])
            if not self.ismissing(value):
                weight = self.ComputeSpatialWeight(dis)
                candidates[value] = candidates.get(value, 0.0) + weight
        return self._candidate_average(candidates)

    def SES(self, ti, tj, dataMatrix):
        candidates = {}
        for span in range(1, self.temporal_threshold + 1):
            ii = ti - span
            if ii >= 0:
                value = float(dataMatrix[ii, tj])
                if not self.ismissing(value):
                    weight = self.ComputeTemporalWeight(span)
                    candidates[value] = candidates.get(value, 0.0) + weight
            ii = ti + span
            if ii < self.rowCount:
                value = float(dataMatrix[ii, tj])
                if not self.ismissing(value):
                    weight = self.ComputeTemporalWeight(span)
                    candidates[value] = candidates.get(value, 0.0) + weight
        return self._candidate_average(candidates)

    def calUserEuclideanSim(self, tj, jj, ti, dataMatrix):
        offset, count = 0.0, 0
        halfWindow = self.windowSize // 2
        upRow = max(ti - halfWindow, 0)
        downRow = min(ti + halfWindow, self.rowCount - 1)
        # C++ uses '< downRow', not '<= downRow'.
        for i in range(upRow, downRow):
            x, y = dataMatrix[i, tj], dataMatrix[i, jj]
            if not self.ismissing(x) and not self.ismissing(y):
                offset += (float(x) - float(y)) ** 2
                count += 1
        if count > 0 and offset > 0:
            return _div(1.0, math.sqrt(offset) / count)
        return 0.0

    def UCF(self, ti, tj, dataMatrix):
        candidates = {}
        for jj in self.stationDic[tj]:
            val = float(dataMatrix[ti, jj])
            if not self.ismissing(val):
                sim = self.calUserEuclideanSim(tj, jj, ti, dataMatrix)
                if sim != 0:
                    candidates[val] = candidates.get(val, 0.0) + sim
        return self._candidate_average(candidates)

    def calItemEuclideanSim(self, ti, ii, tj, dataMatrix):
        offset, count = 0.0, 0
        for jj in self.stationDic[tj]:
            x, y = dataMatrix[ti, jj], dataMatrix[ii, jj]
            if not self.ismissing(x) and not self.ismissing(y):
                offset += (float(x) - float(y)) ** 2
                count += 1
        if count > 0 and offset > 0:
            return _div(1.0, math.sqrt(offset) / count)
        return 0.0

    def ICF(self, ti, tj, dataMatrix):
        candidates = {}
        halfWindow = self.windowSize // 2
        upRow = max(ti - halfWindow, 0)
        downRow = min(ti + halfWindow, self.rowCount - 1)
        for ii in range(upRow, downRow):
            if ii == ti:
                continue
            value = float(dataMatrix[ii, tj])
            if not self.ismissing(value):
                sim = self.calItemEuclideanSim(ti, ii, tj, dataMatrix)
                if sim != 0:
                    candidates[value] = candidates.get(value, 0.0) + sim
        return self._candidate_average(candidates)

    def GlobalViewCombine(self, i, j):
        result_IDW = self.IDW(i, j, self.missingMatrix)
        result_SES = self.SES(i, j, self.missingMatrix)
        if not self.ismissing(result_IDW) and not self.ismissing(result_SES):
            self.temporaryMatrix[i, j] = (result_IDW + result_SES) / 2.0
        elif not self.ismissing(result_SES):
            self.temporaryMatrix[i, j] = result_SES
        elif not self.ismissing(result_IDW):
            self.temporaryMatrix[i, j] = result_IDW

    def InitializeMissing(self):
        for i in range(self.rowCount):
            for j in range(self.columnCount):
                if self.ismissing(self.missingMatrix[i, j]):
                    self.GlobalViewCombine(i, j)

    def checkContextData(self, ti, tj):
        count = 0
        for j in range(self.columnCount):
            if self.ismissing(self.missingMatrix[ti, j]):
                count += 1
        if count > self.columnCount // 2:
            return 0
        halfWindow = self.windowSize // 2
        si = max(ti - halfWindow, 0)
        ei = min(ti + halfWindow, self.rowCount - 1)
        count = 0
        for i in range(si, ei + 1):
            if self.ismissing(self.missingMatrix[i, tj]):
                count += 1
        if count > (ei - si + 1) // 2:
            return 0
        return 1

    def outputCase(self, i, j, trainingMatrix, position):
        ucf = self.UCF(i, j, self.temporaryMatrix)
        icf = self.ICF(i, j, self.temporaryMatrix)
        idw = self.IDW(i, j, self.temporaryMatrix)
        ses = self.SES(i, j, self.temporaryMatrix)
        if not any(self.ismissing(val) for val in (ucf, icf, idw, ses)):
            trainingMatrix[position, :] = [self.missingMatrix[i, j], ucf, idw, icf, ses]
        # The C++ retains the all-zero row if any view is missing.

    def GenerateTrainingCase(self):
        trainingMatrices = []
        for j in range(self.columnCount):
            caseCount = 0
            for i in range(self.rowCount):
                if not self.ismissing(self.missingMatrix[i, j]):
                    if self.checkContextData(i, j) == 1:
                        caseCount += 1
            training = np.zeros((caseCount, self.viewCount + 1), dtype=np.float64)
            counter = 0
            for i in range(self.rowCount):
                if not self.ismissing(self.missingMatrix[i, j]):
                    if self.checkContextData(i, j) == 1:
                        self.outputCase(i, j, training, counter)
                        counter += 1
            trainingMatrices.append(training)
        return trainingMatrices

    @staticmethod
    def chlk(a, n, m, d):
        """Literal translation of the custom in-place C++ Cholesky solver.

        `a` is the original flattened n*n buffer, `d` has length n*m.
        Returns -2 on failed positive-definiteness check and leaves the
        partially modified right-hand side exactly as the C++ would.
        """
        if a[0] + 1.0 == 1.0 or a[0] < 0.0:
            return -2
        a[0] = _sqrt(a[0])
        for j in range(1, n):
            a[j] = _div(a[j], a[0])
        for i in range(1, n):
            u = i * n + i
            for j in range(1, i + 1):
                v = (j - 1) * n + i
                a[u] -= a[v] * a[v]
            if a[u] + 1.0 == 1.0 or a[u] < 0.0:
                return -2
            a[u] = _sqrt(a[u])
            if i != n - 1:
                for j in range(i + 1, n):
                    v = i * n + j
                    for k in range(1, i + 1):
                        a[v] -= a[(k - 1) * n + i] * a[(k - 1) * n + j]
                    a[v] = _div(a[v], a[u])
        for j in range(m):
            d[j] = _div(d[j], a[0])
            for i in range(1, n):
                u = i * n + i
                v = i * m + j
                for k in range(1, i + 1):
                    d[v] -= a[(k - 1) * n + i] * d[(k - 1) * m + j]
                d[v] = _div(d[v], a[u])
        for j in range(m):
            u = (n - 1) * m + j
            d[u] = _div(d[u], a[n * n - 1])
            for k in range(n - 1, 0, -1):
                u = (k - 1) * m + j
                for i in range(k, n):
                    v = (k - 1) * n + i
                    d[u] -= a[v] * d[i * m + j]
                v = (k - 1) * n + k - 1
                d[u] = _div(d[u], a[v])
        return 2

    def sqt2(self, x, y, a, dt, v):
        m, n = x.shape  # x has 4 rows, one column per training case
        mm = m + 1
        b = np.zeros(mm * mm, dtype=np.float64)
        b[mm * mm - 1] = float(n)
        for j in range(m):
            p = 0.0
            for i in range(n):
                p += x[j, i]
            b[m * mm + j] = p
            b[j * mm + m] = p
        for i in range(m):
            for j in range(i, m):
                p = 0.0
                for k in range(n):
                    p += x[i, k] * x[j, k]
                b[j * mm + i] = p
                b[i * mm + j] = p
        a[m] = 0.0
        for i in range(n):
            a[m] += y[i]
        for i in range(m):
            a[i] = 0.0
            for j in range(n):
                a[i] += x[i, j] * y[j]
        # C++ ignores the return code even if Cholesky fails.
        self.chlk(b, mm, 1, a)
        yy = 0.0
        for i in range(n):
            yy += _div(y[i], float(n))
        q = e = u = 0.0
        for i in range(n):
            p = a[m]
            for j in range(m):
                p += a[j] * x[j, i]
            q += (y[i] - p) ** 2
            e += (y[i] - yy) ** 2
            u += (yy - p) ** 2
        s = _sqrt(_div(q, n))
        r = _sqrt(1.0 - _div(q, e))
        for j in range(m):
            p = 0.0
            for i in range(n):
                pp = a[m]
                for k in range(m):
                    if k != j:
                        pp += a[k] * x[k, i]
                p += (y[i] - pp) ** 2
            v[j] = _sqrt(1.0 - _div(q, p))
        dt[:] = [q, s, r, u]

    def FourView(self, sensorCount, trainingMatrices):
        equation = np.zeros((sensorCount, self.viewCount + 1))
        for j in range(sensorCount):
            a = np.zeros(self.viewCount + 1)
            v = np.zeros(self.viewCount)
            dt = np.zeros(4)
            x = trainingMatrices[j][:, 1:].T.copy()
            y = trainingMatrices[j][:, 0].copy()
            self.sqt2(x, y, a, dt, v)
            equation[j, :] = a
        return equation

    def MVL(self, i, j, equation):
        ucf = self.UCF(i, j, self.temporaryMatrix)
        icf = self.ICF(i, j, self.temporaryMatrix)
        idw = self.IDW(i, j, self.temporaryMatrix)
        ses = self.SES(i, j, self.temporaryMatrix)
        if not any(self.ismissing(val) for val in (ucf, icf, idw, ses)):
            self.predictMatrix[i, j] = (equation[j, 0] * ucf + equation[j, 1] * idw
                                        + equation[j, 2] * icf + equation[j, 3] * ses
                                        + equation[j, 4])
        else:
            self.predictMatrix[i, j] = self.temporaryMatrix[i, j]

    def doSTMVL(self, equation):
        for i in range(self.rowCount):
            for j in range(self.columnCount):
                if self.ismissing(self.missingMatrix[i, j]):
                    self.MVL(i, j, equation)
        self.missingMatrix = self.predictMatrix.copy()

    def Run(self):
        self.InitializeMissing()
        trainingMatrices = self.GenerateTrainingCase()
        equation = self.FourView(self.columnCount, trainingMatrices)
        self.doSTMVL(equation)
        return self.missingMatrix


def stmvl_imputation_parametrized(matrix, window_size, gamma, alpha):
    """C++-compatible parameter order; returns a new matrix, not an in-place buffer."""
    model = ST_MVL(matrix, alpha, gamma, window_size)
    recovered = model.Run()
    recovered[np.isnan(recovered)] = math.sqrt(np.finfo(np.float64).max / 100000.0)
    return recovered


def recoverySTMVL(matrix, window_size, gamma, alpha):
    """Convenience alias for the exported C++ recovery operation."""
    return stmvl_imputation_parametrized(matrix, window_size, gamma, alpha)
