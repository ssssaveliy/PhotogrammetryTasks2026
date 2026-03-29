#include "resection.h"

#include <Eigen/SVD>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "sfm_utils.h"
#include "defines.h"

namespace {

    // Сделать из первого минора 3х3 матрицу вращения, скомпенсировать масштаб у компоненты сдвига
    matrix34d canonicalizeP(const matrix34d &P)
    {
        matrix3d RR = P.get_minor<3, 3>(0, 0);
        vector3d tt;
        tt[0] = P(0, 3);
        tt[1] = P(1, 3);
        tt[2] = P(2, 3);

        if (cv::determinant(RR) < 0) {
            RR *= -1;
            tt *= -1;
        }

        double sc = 0;
        for (int i = 0; i < 9; i++) {
            sc += RR.val[i] * RR.val[i];
        }
        sc = std::sqrt(3 / sc);

        Eigen::MatrixXd RRe;
        copy(RR, RRe);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(RRe, Eigen::ComputeFullU | Eigen::ComputeFullV);
        RRe = svd.matrixU() * svd.matrixV().transpose();
        copy(RRe, RR);

        tt *= sc;

        matrix34d result;
        for (int i = 0; i < 9; ++i) {
            result(i / 3, i % 3) = RR(i / 3, i % 3);
        }
        result(0, 3) = tt(0);
        result(1, 3) = tt(1);
        result(2, 3) = tt(2);

        return result;
    }

    cv::Vec2d projectPixel(const phg::Calibration &calib, const cv::Matx34d &P, const cv::Vec3d &X)
    {
        const cv::Vec4d Xh(X[0], X[1], X[2], 1.0);
        const cv::Vec3d cam = P * Xh;
        if (cam[2] == 0.0) {
            throw std::runtime_error("projectPixel: point at infinity");
        }
        const cv::Vec3d pxh = calib.project(cam);
        if (pxh[2] == 0.0) {
            throw std::runtime_error("projectPixel: invalid homogeneous pixel");
        }
        return cv::Vec2d(pxh[0] / pxh[2], pxh[1] / pxh[2]);
    }

    // (см. Hartley & Zisserman p.178)
    cv::Matx34d estimateCameraMatrixDLT(const cv::Vec3d *Xs, const cv::Vec3d *xs, int count)
    {
        if (count < 6) {
            throw std::runtime_error("estimateCameraMatrixDLT: count < 6");
        }

        using mat = Eigen::MatrixXd;
        using vec = Eigen::VectorXd;

        mat A(2 * count, 12);

        for (int i = 0; i < count; ++i) {
            const double x = xs[i][0];
            const double y = xs[i][1];
            const double w = xs[i][2];

            const double X = Xs[i][0];
            const double Y = Xs[i][1];
            const double Z = Xs[i][2];
            const double W = 1.0;

            A(2 * i + 0, 0) = 0.0;
            A(2 * i + 0, 1) = 0.0;
            A(2 * i + 0, 2) = 0.0;
            A(2 * i + 0, 3) = 0.0;
            A(2 * i + 0, 4) = -w * X;
            A(2 * i + 0, 5) = -w * Y;
            A(2 * i + 0, 6) = -w * Z;
            A(2 * i + 0, 7) = -w * W;
            A(2 * i + 0, 8) = y * X;
            A(2 * i + 0, 9) = y * Y;
            A(2 * i + 0, 10) = y * Z;
            A(2 * i + 0, 11) = y * W;

            A(2 * i + 1, 0) = w * X;
            A(2 * i + 1, 1) = w * Y;
            A(2 * i + 1, 2) = w * Z;
            A(2 * i + 1, 3) = w * W;
            A(2 * i + 1, 4) = 0.0;
            A(2 * i + 1, 5) = 0.0;
            A(2 * i + 1, 6) = 0.0;
            A(2 * i + 1, 7) = 0.0;
            A(2 * i + 1, 8) = -x * X;
            A(2 * i + 1, 9) = -x * Y;
            A(2 * i + 1, 10) = -x * Z;
            A(2 * i + 1, 11) = -x * W;
        }

        Eigen::JacobiSVD<mat> svd(A, Eigen::ComputeFullV);
        vec p = svd.matrixV().col(11);

        matrix34d result;
        for (int i = 0; i < 12; ++i) {
            result(i / 4, i % 4) = p[i];
        }

        return canonicalizeP(result);
    }


    // По трехмерным точкам и их проекциям на изображении определяем положение камеры
    cv::Matx34d estimateCameraMatrixRANSAC(const phg::Calibration &calib, const std::vector<cv::Vec3d> &X, const std::vector<cv::Vec2d> &x)
    {
        if (X.size() != x.size()) {
            throw std::runtime_error("estimateCameraMatrixRANSAC: X.size() != x.size()");
        }
        if (X.size() < 6) {
            throw std::runtime_error("estimateCameraMatrixRANSAC: need at least 6 correspondences");
        }

        const int n_points = static_cast<int>(X.size());

        // https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
        // будет отличаться от случая с гомографией
        const int n_trials = std::max(1000, 20 * n_points);

        const double threshold_px = 3;

        const int n_samples = 6;
        uint64_t seed = 1;

        int best_support = 0;
        cv::Matx34d best_P = cv::Matx34d::eye();

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_points, n_samples, &seed);

            cv::Vec3d ms0[n_samples];
            cv::Vec3d ms1[n_samples];
            for (int i = 0; i < n_samples; ++i) {
                ms0[i] = X[sample[i]];
                ms1[i] = calib.unproject(x[sample[i]]);
            }

            cv::Matx34d P = estimateCameraMatrixDLT(ms0, ms1, n_samples);

            int support = 0;
            for (int i = 0; i < n_points; ++i) {
                try {
                    cv::Vec2d px = projectPixel(calib, P, X[i]);
                    if (cv::norm(px - x[i]) < threshold_px) {
                        ++support;
                    }
                } catch (const std::exception &) {
                }
            }

            if (support > best_support) {
                best_support = support;
                best_P = P;

                std::cout << "estimateCameraMatrixRANSAC : support: " << best_support << "/" << n_points << std::endl;

                if (best_support == n_points) {
                    break;
                }
            }
        }

        std::cout << "estimateCameraMatrixRANSAC : best support: " << best_support << "/" << n_points << std::endl;

        if (best_support < 6) {
            throw std::runtime_error("estimateCameraMatrixRANSAC : failed to estimate camera matrix");
        }

        std::vector<cv::Vec3d> inlierX;
        std::vector<cv::Vec3d> inlierx;
        inlierX.reserve(best_support);
        inlierx.reserve(best_support);
        for (int i = 0; i < n_points; ++i) {
            try {
                cv::Vec2d px = projectPixel(calib, best_P, X[i]);
                if (cv::norm(px - x[i]) < threshold_px) {
                    inlierX.push_back(X[i]);
                    inlierx.push_back(calib.unproject(x[i]));
                }
            } catch (const std::exception &) {
            }
        }

        if (static_cast<int>(inlierX.size()) >= 6) {
            best_P = estimateCameraMatrixDLT(inlierX.data(), inlierx.data(), static_cast<int>(inlierX.size()));
        }

        return best_P;
    }


}

cv::Matx34d phg::findCameraMatrix(const Calibration &calib, const std::vector <cv::Vec3d> &X, const std::vector <cv::Vec2d> &x) {
    return estimateCameraMatrixRANSAC(calib, X, x);
}