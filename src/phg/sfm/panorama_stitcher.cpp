#include "panorama_stitcher.h"
#include "homography.h"

#include <libutils/bbox2.h>
#include <iostream>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <vector>

cv::Mat phg::stitchPanorama(const std::vector<cv::Mat> &imgs,
                            const std::vector<int> &parent,
                            std::function<cv::Mat(const cv::Mat &, const cv::Mat &)> &homography_builder)
{
    const int n_images = (int)imgs.size();

    std::vector<cv::Mat> Hs(n_images);
    {
        if ((int)parent.size() != n_images) {
            throw std::runtime_error("stitchPanorama: parent.size() != imgs.size()");
        }

        int root = -1;
        for (int i = 0; i < n_images; ++i) {
            if (parent[i] == -1) {
                if (root != -1) {
                    throw std::runtime_error("stitchPanorama: multiple roots");
                }
                root = i;
            } else {
                if (parent[i] < 0 || parent[i] >= n_images) {
                    throw std::runtime_error("stitchPanorama: invalid parent index");
                }
            }
        }
        if (root == -1) {
            throw std::runtime_error("stitchPanorama: no root image (parent==-1)");
        }

        std::vector<std::vector<int>> children((size_t)n_images);
        children.assign((size_t)n_images, {});
        for (int i = 0; i < n_images; ++i) {
            const int p = parent[i];
            if (p >= 0) {
                children[(size_t)p].push_back(i);
            }
        }

        Hs[root] = cv::Mat::eye(3, 3, CV_64FC1);

        std::vector<char> visited((size_t)n_images, 0);
        visited[(size_t)root] = 1;

        std::vector<int> stack;
        stack.push_back(root);

        while (!stack.empty()) {
            const int cur = stack.back();
            stack.pop_back();

            for (int child : children[(size_t)cur]) {
                cv::Mat H_child_to_parent = homography_builder(imgs[child], imgs[cur]);
                if (H_child_to_parent.empty() || H_child_to_parent.rows != 3 || H_child_to_parent.cols != 3) {
                    throw std::runtime_error("stitchPanorama: homography_builder returned invalid matrix");
                }
                if (H_child_to_parent.type() != CV_64FC1) {
                    H_child_to_parent.convertTo(H_child_to_parent, CV_64F);
                }

                Hs[child] = Hs[cur] * H_child_to_parent;

                visited[(size_t)child] = 1;
                stack.push_back(child);
            }
        }

        for (int i = 0; i < n_images; ++i) {
            if (!visited[(size_t)i]) {
                throw std::runtime_error("stitchPanorama: parent array does not form a connected tree");
            }
        }
    }

    bbox2<double, cv::Point2d> bbox;
    for (int i = 0; i < n_images; ++i) {
        double w = imgs[i].cols;
        double h = imgs[i].rows;
        bbox.grow(phg::transformPoint(cv::Point2d(0.0, 0.0), Hs[i]));
        bbox.grow(phg::transformPoint(cv::Point2d(w, 0.0), Hs[i]));
        bbox.grow(phg::transformPoint(cv::Point2d(w, h), Hs[i]));
        bbox.grow(phg::transformPoint(cv::Point2d(0, h), Hs[i]));
    }

    std::cout << "bbox: " << bbox.max() << ", " << bbox.min() << std::endl;

    int result_width = bbox.width() + 1;
    int result_height = bbox.height() + 1;

    cv::Mat result = cv::Mat::zeros(result_height, result_width, CV_8UC3);

    std::vector<cv::Mat> Hs_inv;
    std::transform(Hs.begin(), Hs.end(), std::back_inserter(Hs_inv), [&](const cv::Mat &H){ return H.inv(); });

#pragma omp parallel for
    for (int y = 0; y < result_height; ++y) {
        for (int x = 0; x < result_width; ++x) {

            cv::Point2d pt_dst(x, y);

            for (int i = 0; i < n_images; ++i) {

                cv::Point2d pt_src = phg::transformPoint(pt_dst + bbox.min(), Hs_inv[i]);

                int x_src = std::round(pt_src.x);
                int y_src = std::round(pt_src.y);

                if (x_src >= 0 && x_src < imgs[i].cols && y_src >= 0 && y_src < imgs[i].rows) {
                    result.at<cv::Vec3b>(y, x) = imgs[i].at<cv::Vec3b>(y_src, x_src);
                    break;
                }
            }

        }
    }

    return result;
}
