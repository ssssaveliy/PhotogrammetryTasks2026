#include "homography.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

    // источник: https://e-maxx.ru/algo/linear_systems_gauss
    // очень важно при выполнении метода гаусса использовать выбор опорного элемента: об этом можно почитать в источнике кода
    // или на вики: https://en.wikipedia.org/wiki/Pivot_element
    int gauss(std::vector<std::vector<double>> a, std::vector<double> &ans)
    {
        using namespace std;
        const double EPS = 1e-8;
        const int INF = std::numeric_limits<int>::max();

        int n = (int) a.size();
        int m = (int) a[0].size() - 1;

        vector<int> where (m, -1);
        for (int col=0, row=0; col<m && row<n; ++col) {
            int sel = row;
            for (int i=row; i<n; ++i)
                if (abs (a[i][col]) > abs (a[sel][col]))
                    sel = i;
            if (abs (a[sel][col]) < EPS)
                continue;
            for (int i=col; i<=m; ++i)
                swap (a[sel][i], a[row][i]);
            where[col] = row;

            for (int i=0; i<n; ++i)
                if (i != row) {
                    double c = a[i][col] / a[row][col];
                    for (int j=col; j<=m; ++j)
                        a[i][j] -= a[row][j] * c;
                }
            ++row;
        }

        ans.assign (m, 0);
        for (int i=0; i<m; ++i)
            if (where[i] != -1)
                ans[i] = a[where[i]][m] / a[where[i]][i];
        for (int i=0; i<n; ++i) {
            double sum = 0;
            for (int j=0; j<m; ++j)
                sum += ans[j] * a[i][j];
            if (abs (sum - a[i][m]) > EPS)
                return 0;
        }

        for (int i=0; i<m; ++i)
            if (where[i] == -1)
                return INF;
        return 1;
    }

    cv::Mat estimateHomography4Points(const cv::Point2f &l0, const cv::Point2f &l1,
                                      const cv::Point2f &l2, const cv::Point2f &l3,
                                      const cv::Point2f &r0, const cv::Point2f &r1,
                                      const cv::Point2f &r2, const cv::Point2f &r3)
    {
        std::vector<std::vector<double>> A;
        std::vector<double> H;

        double xs0[4] = {l0.x, l1.x, l2.x, l3.x};
        double xs1[4] = {r0.x, r1.x, r2.x, r3.x};
        double ys0[4] = {l0.y, l1.y, l2.y, l3.y};
        double ys1[4] = {r0.y, r1.y, r2.y, r3.y};
        double ws0[4] = {1, 1, 1, 1};
        double ws1[4] = {1, 1, 1, 1};

        for (int i = 0; i < 4; ++i) {
            double x0 = xs0[i];
            double y0 = ys0[i];
            double w0 = ws0[i];

            double x1 = xs1[i];
            double y1 = ys1[i];
            double w1 = ws1[i];

            (void)w1;

            A.push_back({
                x0, y0, w0, 0.0, 0.0, 0.0,
                -x1 * x0, -x1 * y0,
                x1 * w0
            });
            A.push_back({
                0.0, 0.0, 0.0, x0, y0, w0,
                -y1 * x0, -y1 * y0,
                y1 * w0
            });
        }

        int res = gauss(A, H);
        if (res == 0) {
            throw std::runtime_error("gauss: no solution found");
        } else if (res == 1) {
            // ok
        } else if (res == std::numeric_limits<int>::max()) {
            std::cerr << "gauss: infinitely many solutions found" << std::endl;
        } else {
            throw std::runtime_error("gauss: unexpected return code");
        }

        H.push_back(1.0);

        cv::Mat H_mat(3, 3, CV_64FC1);
        std::copy(H.begin(), H.end(), H_mat.ptr<double>());
        return H_mat;
    }

    inline uint64_t xorshift64(uint64_t *state)
    {
        if (*state == 0) {
            *state = 1;
        }

        uint64_t x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return *state = x;
    }

    void randomSample(std::vector<int> &dst, int max_id, int sample_size, uint64_t *state)
    {
        dst.clear();
        const int max_attempts = 1000;

        for (int i = 0; i < sample_size; ++i) {
            for (int k = 0; k < max_attempts; ++k) {
                int v = (int)(xorshift64(state) % (uint64_t)max_id);
                if (dst.empty() || std::find(dst.begin(), dst.end(), v) == dst.end()) {
                    dst.push_back(v);
                    break;
                }
            }
            if ((int)dst.size() < i + 1) {
                throw std::runtime_error("Failed to sample ids");
            }
        }
    }

    cv::Mat estimateHomographyRANSAC(const std::vector<cv::Point2f> &points_lhs,
                                     const std::vector<cv::Point2f> &points_rhs)
    {
        if (points_lhs.size() != points_rhs.size()) {
            throw std::runtime_error("findHomography: points_lhs.size() != points_rhs.size()");
        }
        if (points_lhs.size() < 4) {
            throw std::runtime_error("estimateHomographyRANSAC : too few points");
        }

        const int n_matches = (int)points_lhs.size();

        const double reprojection_error_threshold_px = 3.0;
        const double reprojection_error_threshold2 = reprojection_error_threshold_px * reprojection_error_threshold_px;
        const double confidence = 0.999;
        const int sample_size = 4;
        int max_trials = 2000;

        auto isDegenerate = [&](const std::vector<int> &s) -> bool {
            auto area2 = [](const cv::Point2f &a, const cv::Point2f &b, const cv::Point2f &c) {
                cv::Point2f ab = b - a;
                cv::Point2f ac = c - a;
                return std::abs(ab.x * ac.y - ab.y * ac.x);
            };
            const double eps = 1e-3;

            const cv::Point2f &p0 = points_lhs[s[0]];
            const cv::Point2f &p1 = points_lhs[s[1]];
            const cv::Point2f &p2 = points_lhs[s[2]];
            const cv::Point2f &p3 = points_lhs[s[3]];
            const cv::Point2f &q0 = points_rhs[s[0]];
            const cv::Point2f &q1 = points_rhs[s[1]];
            const cv::Point2f &q2 = points_rhs[s[2]];
            const cv::Point2f &q3 = points_rhs[s[3]];

            if (area2(p0, p1, p2) < eps) return true;
            if (area2(p0, p1, p3) < eps) return true;
            if (area2(p0, p2, p3) < eps) return true;
            if (area2(p1, p2, p3) < eps) return true;

            if (area2(q0, q1, q2) < eps) return true;
            if (area2(q0, q1, q3) < eps) return true;
            if (area2(q0, q2, q3) < eps) return true;
            if (area2(q1, q2, q3) < eps) return true;

            return false;
        };

        uint64_t seed = 1;
        int best_support = 0;
        cv::Mat best_H;
        std::vector<char> best_inliers_mask(n_matches, 0);

        int trials = max_trials;
        std::vector<int> sample;
        std::vector<char> inliers_mask(n_matches);

        for (int i_trial = 0; i_trial < trials; ++i_trial) {
            randomSample(sample, n_matches, sample_size, &seed);
            if (isDegenerate(sample)) {
                continue;
            }

            cv::Mat H;
            try {
                H = estimateHomography4Points(points_lhs[sample[0]], points_lhs[sample[1]], points_lhs[sample[2]], points_lhs[sample[3]],
                                              points_rhs[sample[0]], points_rhs[sample[1]], points_rhs[sample[2]], points_rhs[sample[3]]);
            } catch (...) {
                continue;
            }

            int support = 0;
            for (int i_point = 0; i_point < n_matches; ++i_point) {
                bool inl = false;
                try {
                    cv::Point2d proj = phg::transformPoint(points_lhs[i_point], H);
                    cv::Point2d diff = proj - cv::Point2d(points_rhs[i_point]);
                    double d2 = diff.x * diff.x + diff.y * diff.y;
                    inl = (d2 <= reprojection_error_threshold2);
                } catch (...) {
                    inl = false;
                }

                inliers_mask[i_point] = (char)inl;
                if (inl) ++support;
            }

            if (support > best_support) {
                best_support = support;
                best_H = H;
                best_inliers_mask = inliers_mask;

                double w = (double)best_support / (double)n_matches;
                w = std::clamp(w, 1e-6, 1.0 - 1e-6);
                double p_no_outliers = std::pow(w, sample_size);
                p_no_outliers = std::clamp(p_no_outliers, 1e-12, 1.0 - 1e-12);
                double new_trials = std::log(1.0 - confidence) / std::log(1.0 - p_no_outliers);
                if (std::isfinite(new_trials)) {
                    trials = std::min(trials, (int)std::ceil(new_trials));
                    trials = std::clamp(trials, 50, max_trials);
                }

                if (best_support == n_matches) {
                    break;
                }
            }
        }

        if (best_support < 4) {
            throw std::runtime_error("estimateHomographyRANSAC : failed to estimate homography");
        }

        std::vector<cv::Point2f> inl_lhs;
        std::vector<cv::Point2f> inl_rhs;
        inl_lhs.reserve((size_t)best_support);
        inl_rhs.reserve((size_t)best_support);

        for (int i = 0; i < n_matches; ++i) {
            if (best_inliers_mask[i]) {
                inl_lhs.push_back(points_lhs[i]);
                inl_rhs.push_back(points_rhs[i]);
            }
        }

        if (inl_lhs.size() >= 4) {
            cv::Mat refined = cv::findHomography(inl_lhs, inl_rhs, 0);
            if (!refined.empty()) {
                double s = refined.at<double>(2, 2);
                if (std::abs(s) > 1e-12) {
                    refined /= s;
                }
                return refined;
            }
        }

        return best_H;
    }

}

cv::Mat phg::findHomography(const std::vector<cv::Point2f> &points_lhs, const std::vector<cv::Point2f> &points_rhs)
{
    return estimateHomographyRANSAC(points_lhs, points_rhs);
}

cv::Mat phg::findHomographyCV(const std::vector<cv::Point2f> &points_lhs, const std::vector<cv::Point2f> &points_rhs)
{
    return cv::findHomography(points_lhs, points_rhs, cv::RANSAC);
}

cv::Point2d phg::transformPoint(const cv::Point2d &pt, const cv::Mat &T)
{
    if (T.empty() || T.rows != 3 || T.cols != 3) {
        throw std::runtime_error("transformPoint: expected 3x3 matrix");
    }

    double a00, a01, a02;
    double a10, a11, a12;
    double a20, a21, a22;

    if (T.type() == CV_64FC1) {
        a00 = T.at<double>(0, 0); a01 = T.at<double>(0, 1); a02 = T.at<double>(0, 2);
        a10 = T.at<double>(1, 0); a11 = T.at<double>(1, 1); a12 = T.at<double>(1, 2);
        a20 = T.at<double>(2, 0); a21 = T.at<double>(2, 1); a22 = T.at<double>(2, 2);
    } else if (T.type() == CV_32FC1) {
        a00 = T.at<float>(0, 0); a01 = T.at<float>(0, 1); a02 = T.at<float>(0, 2);
        a10 = T.at<float>(1, 0); a11 = T.at<float>(1, 1); a12 = T.at<float>(1, 2);
        a20 = T.at<float>(2, 0); a21 = T.at<float>(2, 1); a22 = T.at<float>(2, 2);
    } else {
        cv::Mat Td;
        T.convertTo(Td, CV_64F);
        a00 = Td.at<double>(0, 0); a01 = Td.at<double>(0, 1); a02 = Td.at<double>(0, 2);
        a10 = Td.at<double>(1, 0); a11 = Td.at<double>(1, 1); a12 = Td.at<double>(1, 2);
        a20 = Td.at<double>(2, 0); a21 = Td.at<double>(2, 1); a22 = Td.at<double>(2, 2);
    }

    const double x = pt.x;
    const double y = pt.y;

    const double w = a20 * x + a21 * y + a22;
    if (!std::isfinite(w) || std::abs(w) < 1e-12) {
        throw std::runtime_error("transformPoint: invalid homogeneous coordinate");
    }

    const double xp = (a00 * x + a01 * y + a02) / w;
    const double yp = (a10 * x + a11 * y + a12) / w;
    return {xp, yp};
}

cv::Point2d phg::transformPointCV(const cv::Point2d &pt, const cv::Mat &T) {
    std::vector<cv::Point2f> tmp0 = {pt};
    std::vector<cv::Point2f> tmp1(1);
    cv::perspectiveTransform(tmp0, tmp1, T);
    return tmp1[0];
}
