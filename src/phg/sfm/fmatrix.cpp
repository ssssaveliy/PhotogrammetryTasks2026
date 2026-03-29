#include "fmatrix.h"
#include "sfm_utils.h"
#include "defines.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <Eigen/SVD>
#include <opencv2/calib3d.hpp>

namespace {

    void infoF(const cv::Matx33d &Fcv)
    {
        Eigen::MatrixXd F;
        copy(Fcv, F);

        Eigen::JacobiSVD<Eigen::MatrixXd> svdf(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::MatrixXd U = svdf.matrixU();
        Eigen::VectorXd s = svdf.singularValues();
        Eigen::MatrixXd V = svdf.matrixV();

        std::cout << "F info:\nF:\n" << F << "\nU:\n" << U << "\ns:\n" << s << "\nV:\n" << V << std::endl;
    }

    cv::Matx33d mul(const cv::Matx33d &A, const cv::Matx33d &B)
    {
        return A * B;
    }

    // (см. Hartley & Zisserman p.279)
    cv::Matx33d estimateFMatrixDLT(const cv::Vec2d *m0, const cv::Vec2d *m1, int count)
    {
        if (count < 8) {
            throw std::runtime_error("estimateFMatrixDLT: count < 8");
        }

        const int a_rows = count;
        const int a_cols = 9;

        Eigen::MatrixXd A(a_rows, a_cols);

        for (int i_pair = 0; i_pair < count; ++i_pair) {
            const double x0 = m0[i_pair][0];
            const double y0 = m0[i_pair][1];

            const double x1 = m1[i_pair][0];
            const double y1 = m1[i_pair][1];

            A(i_pair, 0) = x1 * x0;
            A(i_pair, 1) = x1 * y0;
            A(i_pair, 2) = x1;
            A(i_pair, 3) = y1 * x0;
            A(i_pair, 4) = y1 * y0;
            A(i_pair, 5) = y1;
            A(i_pair, 6) = x0;
            A(i_pair, 7) = y0;
            A(i_pair, 8) = 1.0;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullV);
        Eigen::VectorXd null_space = svda.matrixV().col(8);

        Eigen::MatrixXd F(3, 3);
        F.row(0) << null_space[0], null_space[1], null_space[2];
        F.row(1) << null_space[3], null_space[4], null_space[5];
        F.row(2) << null_space[6], null_space[7], null_space[8];

        // Поправить F так, чтобы соблюдалось свойство фундаментальной матрицы (последнее сингулярное значение = 0)
        Eigen::JacobiSVD<Eigen::MatrixXd> svdf(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd s = svdf.singularValues();
        Eigen::MatrixXd S = Eigen::MatrixXd::Zero(3, 3);
        S(0, 0) = s[0];
        S(1, 1) = s[1];
        F = svdf.matrixU() * S * svdf.matrixV().transpose();

        cv::Matx33d Fcv;
        copy(F, Fcv);

        return Fcv;
    }

    // Нужно создать матрицу преобразования, которая сдвинет переданное множество точек так, что центр масс перейдет в ноль, а Root Mean Square расстояние до него станет sqrt(2)
    // (см. Hartley & Zisserman p.107 Why is normalization essential?)
    cv::Matx33d getNormalizeTransform(const std::vector<cv::Vec2d> &m)
    {
        if (m.empty()) {
            throw std::runtime_error("getNormalizeTransform: empty input");
        }

        cv::Vec2d mean(0.0, 0.0);
        for (const cv::Vec2d &pt : m) {
            mean += pt;
        }
        mean *= (1.0 / static_cast<double>(m.size()));

        double rms2 = 0.0;
        for (const cv::Vec2d &pt : m) {
            const cv::Vec2d d = pt - mean;
            rms2 += d.dot(d);
        }
        rms2 /= static_cast<double>(m.size());
        const double rms = std::sqrt(rms2);

        double s = 1.0;
        if (rms > std::numeric_limits<double>::epsilon()) {
            s = std::sqrt(2.0) / rms;
        }

        return cv::Matx33d(
            s, 0.0, -s * mean[0],
            0.0, s, -s * mean[1],
            0.0, 0.0, 1.0
        );
    }

    cv::Vec2d transformPoint(const cv::Vec2d &pt, const cv::Matx33d &T)
    {
        cv::Vec3d tmp = T * cv::Vec3d(pt[0], pt[1], 1.0);

        if (tmp[2] == 0) {
            throw std::runtime_error("infinite point");
        }

        return cv::Vec2d(tmp[0] / tmp[2], tmp[1] / tmp[2]);
    }

    cv::Matx33d estimateFMatrixNormalizedDLT(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1)
    {
        if (m0.size() != m1.size()) {
            throw std::runtime_error("estimateFMatrixNormalizedDLT: m0.size() != m1.size()");
        }
        if (m0.size() < 8) {
            throw std::runtime_error("estimateFMatrixNormalizedDLT: need at least 8 matches");
        }

        const cv::Matx33d T0 = getNormalizeTransform(m0);
        const cv::Matx33d T1 = getNormalizeTransform(m1);

        std::vector<cv::Vec2d> m0n(m0.size());
        std::vector<cv::Vec2d> m1n(m1.size());
        for (size_t i = 0; i < m0.size(); ++i) {
            m0n[i] = transformPoint(m0[i], T0);
            m1n[i] = transformPoint(m1[i], T1);
        }

        cv::Matx33d F = estimateFMatrixDLT(m0n.data(), m1n.data(), static_cast<int>(m0n.size()));
        F = mul(T1.t(), mul(F, T0));

        return F;
    }

    cv::Matx33d estimateFMatrixRANSAC(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px)
    {
        if (m0.size() != m1.size()) {
            throw std::runtime_error("estimateFMatrixRANSAC: m0.size() != m1.size()");
        }
        if (m0.size() < 8) {
            throw std::runtime_error("estimateFMatrixRANSAC: need at least 8 matches");
        }

        const int n_matches = static_cast<int>(m0.size());

        if (n_matches == 8) {
            return estimateFMatrixNormalizedDLT(m0, m1);
        }

        cv::Matx33d TN0 = getNormalizeTransform(m0);
        cv::Matx33d TN1 = getNormalizeTransform(m1);

        std::vector<cv::Vec2d> m0_t(n_matches);
        std::vector<cv::Vec2d> m1_t(n_matches);
        for (int i = 0; i < n_matches; ++i) {
            m0_t[i] = transformPoint(m0[i], TN0);
            m1_t[i] = transformPoint(m1[i], TN1);
        }

        const int n_trials = std::max(1000, 50 * n_matches);
        const int n_samples = 8;
        uint64_t seed = 1;

        int best_support = 0;
        cv::Matx33d best_F = cv::Matx33d::eye();

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_matches, n_samples, &seed);

            cv::Vec2d ms0[n_samples];
            cv::Vec2d ms1[n_samples];
            for (int i = 0; i < n_samples; ++i) {
                ms0[i] = m0_t[sample[i]];
                ms1[i] = m1_t[sample[i]];
            }

            cv::Matx33d F = estimateFMatrixDLT(ms0, ms1, n_samples);
            F = mul(TN1.t(), mul(F, TN0));

            int support = 0;
            for (int i = 0; i < n_matches; ++i) {
                if (phg::epipolarTest(m0[i], m1[i], F, threshold_px) &&
                    phg::epipolarTest(m1[i], m0[i], F.t(), threshold_px)) {
                    ++support;
                }
            }

            if (support > best_support) {
                best_support = support;
                best_F = F;

                std::cout << "estimateFMatrixRANSAC : support: " << best_support << "/" << n_matches << std::endl;
                infoF(F);

                if (best_support == n_matches) {
                    break;
                }
            }
        }

        std::cout << "estimateFMatrixRANSAC : best support: " << best_support << "/" << n_matches << std::endl;

        if (best_support < 8) {
            return estimateFMatrixNormalizedDLT(m0, m1);
        }

        std::vector<cv::Vec2d> inliers0;
        std::vector<cv::Vec2d> inliers1;
        inliers0.reserve(best_support);
        inliers1.reserve(best_support);
        for (int i = 0; i < n_matches; ++i) {
            if (phg::epipolarTest(m0[i], m1[i], best_F, threshold_px) &&
                phg::epipolarTest(m1[i], m0[i], best_F.t(), threshold_px)) {
                inliers0.push_back(m0[i]);
                inliers1.push_back(m1[i]);
            }
        }

        if (static_cast<int>(inliers0.size()) >= 8) {
            best_F = estimateFMatrixNormalizedDLT(inliers0, inliers1);
        }

        return best_F;
    }

}

cv::Matx33d phg::findFMatrix(const std::vector <cv::Vec2d> &m0, const std::vector <cv::Vec2d> &m1, double threshold_px) {
    return estimateFMatrixRANSAC(m0, m1, threshold_px);
}

cv::Matx33d phg::findFMatrixCV(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px) {
    return cv::findFundamentalMat(m0, m1, cv::FM_RANSAC, threshold_px);
}

cv::Matx33d phg::composeFMatrix(const cv::Matx34d &P0, const cv::Matx34d &P1)
{
    cv::Matx33d F;

#define det4(a, b, c, d) \
      ((a)(0) * (b)(1) - (a)(1) * (b)(0)) * ((c)(2) * (d)(3) - (c)(3) * (d)(2)) - \
      ((a)(0) * (b)(2) - (a)(2) * (b)(0)) * ((c)(1) * (d)(3) - (c)(3) * (d)(1)) + \
      ((a)(0) * (b)(3) - (a)(3) * (b)(0)) * ((c)(1) * (d)(2) - (c)(2) * (d)(1)) + \
      ((a)(1) * (b)(2) - (a)(2) * (b)(1)) * ((c)(0) * (d)(3) - (c)(3) * (d)(0)) - \
      ((a)(1) * (b)(3) - (a)(3) * (b)(1)) * ((c)(0) * (d)(2) - (c)(2) * (d)(0)) + \
      ((a)(2) * (b)(3) - (a)(3) * (b)(2)) * ((c)(0) * (d)(1) - (c)(1) * (d)(0))

    int i, j;
    for (j = 0; j < 3; j++)
        for (i = 0; i < 3; i++) {
            const auto a1 = P0.row((i + 1) % 3);
            const auto a2 = P0.row((i + 2) % 3);
            const auto b1 = P1.row((j + 1) % 3);
            const auto b2 = P1.row((j + 2) % 3);

            F(j, i) = det4(a1, a2, b1, b2);
        }

#undef det4
    
    return F;
}