#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

Mat meanshift(Mat img, float hs, float hc, float eps, int kmax)
{
    Mat imgFloat;
    img.convertTo(imgFloat, CV_32FC3);

    const int H = img.rows;
    const int W = img.cols;

    // Espace 5D : (x, y, R, G, B)
    vector<vector<Vec<float,5>>> X(H, vector<Vec<float,5>>(W));

    for(int i = 0; i < H; i++)
    {
        for(int j = 0; j < W; j++)
        {
            Vec3f color = imgFloat.at<Vec3f>(i,j);
            X[i][j] = { (float)i, (float)j, color[0], color[1], color[2] };
        }
    }

    int k = 1;

    while(k <= kmax)
    {
        float maxShift = 0.0f;

        for(int i = 0; i < H; i++)
        {
            for(int j = 0; j < W; j++)
            {
                Vec<float,5> x = X[i][j];
                Vec<float,5> mean = {0,0,0,0,0};
                int count = 0;

                for(int m = 0; m < H; m++)
                {
                    for(int n = 0; n < W; n++)
                    {
                        Vec<float,5> y = X[m][n];

                        float spatialDist = norm(Vec2f(x[0]-y[0], x[1]-y[1]));
                        float colorDist   = norm(Vec3f(x[2]-y[2], x[3]-y[3], x[4]-y[4]));

                        if(spatialDist < hs && colorDist < hc)
                        {
                            mean += y;
                            count++;
                        }
                    }
                }

                if(count > 0)
                {
                    mean /= count;

                    float shift = norm(mean - x);
                    maxShift = max(maxShift, shift);

                    X[i][j] = mean;
                }
            }
        }

        if(maxShift < eps)
            break;

        k++;
    }

    // Reconstruction image segmentée
    Mat result(H, W, CV_8UC3);

    for(int i = 0; i < H; i++)
    {
        for(int j = 0; j < W; j++)
        {
            Vec<float,5> x = X[i][j];
            result.at<Vec3b>(i,j) = Vec3b(
                (uchar)x[2],
                (uchar)x[3],
                (uchar)x[4]
            );
        }
    }

    return result;
}

int main(int argc, char** argv)
{
    const std::string keys =
        "{help h usage ?     |       | print this message }"
        "{input i            |<none> | input image file   }"
        "{hs                 |10     | spatial bandwidth  }"
        "{hc                 |20     | color bandwidth    }"
        "{epsilon e          |1.0    | convergence threshold }"
        "{kmax               |10     | max iterations     }"
        "{groundtruth gt     |       | ground truth segmentation image (optional) }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("Mean-Shift segmentation application");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    const string imageFilename = parser.get<string>("input");
    const float hs = parser.get<float>("hs");
    const float hc = parser.get<float>("hc");
    const float epsilon = parser.get<float>("epsilon");
    const int kmax = parser.get<int>("kmax");
    const string groundTruthFilename = parser.get<string>("groundtruth");

    if (!parser.check())
    {
        parser.printErrors();
        return EXIT_FAILURE;
    }

    cout << " Program called with:" << endl;
    cout << " \timage file: " << imageFilename << endl;
    cout << " \ths: " << hs << endl;
    cout << " \thc: " << hc << endl;
    cout << " \tepsilon: " << epsilon << endl;
    cout << " \tkmax: " << kmax << endl;

    // ==========================
    // Lecture image
    // ==========================
    Mat img = imread(imageFilename);
    PRINT_MAT_INFO(img);

    // ==========================
    // Mean-Shift perso
    // ==========================
    Mat segmented = meanShiftSegmentation(img, hs, hc, epsilon, kmax);

    PRINT_MAT_INFO(segmented);

    // Conversion binaire simple (comme kmeans)
    Mat segmentedGray;
    cvtColor(segmented, segmentedGray, COLOR_BGR2GRAY);

    Mat mask255;
    threshold(segmentedGray, mask255, 0, 255, THRESH_BINARY | THRESH_OTSU);

    imwrite("../res/mean_shift_result.png", segmented);
    imwrite("../res/mean_shift_mask.png", mask255);

    // ==========================
    // Évaluation si GT fournie
    // ==========================
    if (!groundTruthFilename.empty())
    {
        Mat gt = imread(groundTruthFilename, IMREAD_GRAYSCALE);
        if (gt.empty())
        {
            cerr << "Error loading ground truth." << endl;
            return EXIT_FAILURE;
        }

        Mat pred_pos = (mask255 > 0);
        Mat gt_pos = (gt == 0);

        Mat TPmat = pred_pos & gt_pos;
        Mat FPmat = pred_pos & (~gt_pos);
        Mat FNmat = (~pred_pos) & gt_pos;

        double TP = (double)countNonZero(TPmat);
        double FP = (double)countNonZero(FPmat);
        double FN = (double)countNonZero(FNmat);

        auto div = [](double a, double b) {
            return (b == 0.0) ? 0.0 : a / b;
        };

        double P = div(TP, TP + FP);
        double S = div(TP, TP + FN);
        double DSC = div(2.0 * TP, 2.0 * TP + FP + FN);

        cout << "Mean-Shift Results:" << endl;
        cout << "P  = " << P << endl;
        cout << "S  = " << S << endl;
        cout << "DSC = " << DSC << endl;
    }

    return EXIT_SUCCESS;
}