#include "ocv_utils.hpp"
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <limits>
#include <utility>

using namespace cv;
using namespace std;
using std::filesystem::path;

auto div(double a, double b){
    if (b==0.0){
        return 0.0;
    } else {
        return a/b;
    }
}

auto dist2_5 = [](const float* a, const float* b) -> float {
    float s = 0.f;
    for (int j = 0; j < 5; ++j) {
        float d = a[j] - b[j];
        s += d * d;
    }
    return s;
};

/**
 * K-means "perso" pour segmentation.
 *
 * - img : image d'entrée (CV_8UC1, CV_8UC3, ou autre convertible)
 * - k   : nombre de clusters
 * - iter: nb max d'itérations
 *
 * Retour :
 * - Mat labelsImg (H x W) de type CV_32S, où labelsImg(y,x) ∈ {0,...,k-1}
 *
 * On fait un clustering dans un feature space 5D : (B,G,R, λ*x~, λ*y~) -> permet plus de cohérence spatiale qu'uniquement la couleur.
 * - Pour faire couleur seule : mettre LAMBDA = 0.f.
 */
Mat kmeans_perso(Mat img, int k, int iter){
    const int H = img.rows;
    const int W = img.cols;
    const int N = H * W;

    // Paramètre spatial (à ajuster)
    const float LAMBDA = 120.0f;

    // Préparer sortie labels
    Mat labels(N, 1, CV_32S, cv::Scalar(0));

    // Centres : k x 5 en float
    Mat centers(k, 5, CV_32F);

    Mat bgr8 = img;

    // RNG
    cv::RNG rng((uint64)cv::getTickCount());

    // Initialisation : choisir k pixels aléatoires comme centres init
    for (int c = 0; c < k; ++c) {
        int idx = rng.uniform(0, N);
        int y = idx / W;
        int x = idx - y * W;

        const cv::Vec3b pix = bgr8.at<cv::Vec3b>(y, x);
        float* ctr = centers.ptr<float>(c);
        ctr[0] = (float)pix[0];
        ctr[1] = (float)pix[1];
        ctr[2] = (float)pix[2];
        ctr[3] = LAMBDA * (W > 1 ? (float)x / (float)(W - 1) : 0.0f);
        ctr[4] = LAMBDA * (H > 1 ? (float)y / (float)(H - 1) : 0.0f);
    }

    Mat newCenters(k, 5, CV_32F);
    std::vector<int> counts(k, 0);
    for (int it = 0; it < iter; ++it) {
        bool changed = false;
        //Attribuer un cluster à chaque pixel
        for (int i = 0; i < N; ++i) {
            int y = i / W;
            int x = i - y * W;

            const cv::Vec3b pix = bgr8.at<cv::Vec3b>(y, x);
            //features du pixel
            float f[5];
            f[0] = (float)pix[0];
            f[1] = (float)pix[1];
            f[2] = (float)pix[2];
            f[3] = LAMBDA * (W > 1 ? (float)x / (float)(W - 1) : 0.0f);
            f[4] = LAMBDA * (H > 1 ? (float)y / (float)(H - 1) : 0.0f);

            int best = 0;
            float bestD = std::numeric_limits<float>::infinity();
            
            //trouver le centre le plus proche du pixel
            for (int c = 0; c < k; ++c) {
                const float* ctr = centers.ptr<float>(c);
                float d2 = dist2_5(f, ctr);
                if (d2 < bestD) { 
                    bestD = d2; best = c; 
                }
            }


            //attribuer le centre au pixel
            int& lab = labels.at<int>(i, 0);
            if (lab != best) { 
                lab = best; changed = true; 
            }
        }
        //Recalculer les centres
        newCenters.setTo(0);
        std::fill(counts.begin(), counts.end(), 0);
        //accumuler les features des pixels appartenant à chaque cluster
        for (int i = 0; i < N; ++i) {
            int c = labels.at<int>(i, 0);
            int y = i / W;
            int x = i - y * W;

            const cv::Vec3b pix = bgr8.at<cv::Vec3b>(y, x);

            float* acc = newCenters.ptr<float>(c);
            acc[0] += (float)pix[0];
            acc[1] += (float)pix[1];
            acc[2] += (float)pix[2];
            acc[3] += LAMBDA * (W > 1 ? (float)x / (float)(W - 1) : 0.0f);
            acc[4] += LAMBDA * (H > 1 ? (float)y / (float)(H - 1) : 0.0f);

            counts[c]++;
        }
        //Moyenner pour obtenir les nouveaux centres
        for (int c = 0; c < k; ++c) {
            float* ctr = newCenters.ptr<float>(c);
            if (counts[c] > 0) {
                float inv = 1.f / (float)counts[c];
                for (int j = 0; j < 5; ++j) {
                    ctr[j] *= inv;
                }
            } else {
                int idx = rng.uniform(0, N);
                int y = idx / W;
                int x = idx - y * W;
                const cv::Vec3b pix = bgr8.at<cv::Vec3b>(y, x);

                ctr[0] = (float)pix[0];
                ctr[1] = (float)pix[1];
                ctr[2] = (float)pix[2];
                ctr[3] = LAMBDA * (W > 1 ? (float)x / (float)(W - 1) : 0.0f);
                ctr[4] = LAMBDA * (H > 1 ? (float)y / (float)(H - 1) : 0.0f);
            }
        }

        //si labels n'ont pas changé, on stop
        centers = newCenters.clone();
        if (!changed) break;
    }

    Mat labelsImg = labels.reshape(1, H); // CV_32S, HxW
    return labelsImg;
}

static cv::Mat meanshift(const cv::Mat& imgBGR8, int hs, float hc, float eps, int kmax)
{
    CV_Assert(imgBGR8.type() == CV_8UC3);

    const int H = imgBGR8.rows, W = imgBGR8.cols;
    const float hc2  = hc * hc;
    const float eps2 = eps * eps;

    cv::Mat cur, next;
    imgBGR8.convertTo(cur, CV_32FC3);
    next.create(H, W, CV_32FC3);

    for (int it = 0; it < kmax; ++it) {
        float maxShift2 = 0.f;

        for (int y = 0; y < H; ++y) {
            const cv::Vec3f* curRow = cur.ptr<cv::Vec3f>(y);
            cv::Vec3f* nextRow = next.ptr<cv::Vec3f>(y);

            for (int x = 0; x < W; ++x) {

                const cv::Vec3f c0 = curRow[x];  // couleur courante du pixel central

                const int ymin = std::max(0, y - hs);
                const int ymax = std::min(H - 1, y + hs);
                const int xmin = std::max(0, x - hs);
                const int xmax = std::min(W - 1, x + hs);

                cv::Vec3f sum(0,0,0);
                int count = 0;

                for (int yy = ymin; yy <= ymax; ++yy) {
                    const cv::Vec3f* rowN = cur.ptr<cv::Vec3f>(yy);
                    for (int xx = xmin; xx <= xmax; ++xx) {
                        const cv::Vec3f cN = rowN[xx];

                        cv::Vec3f d = cN - c0;
                        float d2 = d.dot(d);
                        if (d2 <= hc2) {
                            sum += cN;
                            ++count;
                        }
                    }
                }

                cv::Vec3f mean = (count > 0) ? (sum * (1.f / (float)count)) : c0;

                cv::Vec3f diff = mean - c0;
                float shift2 = diff.dot(diff);
                if (shift2 > maxShift2) maxShift2 = shift2;

                nextRow[x] = mean;
            }
        }

        cur = next.clone();              // mise à jour globale
        if (maxShift2 <= eps2) break;    // critère d’arrêt
    }

    // image segmentée (quantifiée) : conversion en 8U pour sauver/debug
    cv::Mat out8;
    cur.convertTo(out8, CV_8UC3);
    return out8;
}

int main(int argc, char** argv)
{
    // ============================================================================
    // CommandLineParser: How to handle command-line arguments
    // ============================================================================
    // When you run a program from the terminal, you can pass arguments to it.
    // For example: ./kmeans -i=cat.jpg -k=5 --gt=ground_truth.jpg
    //
    // The CommandLineParser helps us extract and validate these arguments.
    //
    // HOW THE "keys" STRING WORKS:
    // Each line in curly braces {} defines one parameter with this format:
    // "{name aliases | default_value | description}"
    //
    // - name: The primary name for the parameter (e.g., "input", "k")
    // - aliases: Short versions you can use instead (e.g., "i" for "input")
    // - default_value: What value to use if the user doesn't provide this parameter
    //   * <none> means the parameter is REQUIRED (program will fail without it)
    //   * Empty string means the parameter is OPTIONAL
    //   * Any other value is used as the default
    // - description: Help text shown when user runs the program with --help
    //
    // IMPORTANT: SYNTAX FOR PASSING VALUES
    // OpenCV's CommandLineParser requires an EQUALS SIGN (=) to pass values:
    // ✓ CORRECT:   -i=cat.jpg  or  --input=cat.jpg
    // ✗ WRONG:     -i cat.jpg  (this treats -i as a boolean flag, "cat.jpg" ignored)
    //
    // EXAMPLES OF HOW TO USE THIS PROGRAM:
    // ./kmeans -i=cat.jpg -k=5                          (required parameters only)
    // ./kmeans --input=cat.jpg -k=5 --gt=truth.jpg      (with optional groundtruth)
    // ./kmeans --help                                   (shows help message)
    //
    // RETRIEVING THE VALUES:
    // After defining the parameters, use parser.get<Type>("name") to retrieve them:
    // - parser.get<String>("input") gets the filename as a string
    // - parser.get<int>("k") gets the number of clusters as an integer
    //
    // HOW TO ADD A NEW PARAMETER (Example: number of iterations):
    // 1. Add a new line in the "keys" string:
    //    "{iterations iter  |100    | maximum number of iterations }"
    //    This creates an optional parameter with default value 100
    //
    // 2. Retrieve it after parser.check():
    //    const int iterations = parser.get<int>("iterations");
    //
    // 3. Use it in your program, for example when calling kmeans:
    //    kmeans(data, k, labels, TermCriteria(..., iterations, ...), ...);
    //
    // 4. Users can now run: ./kmeans -i=cat.jpg -k=5 --iterations=200
    //    Remember: use = to pass the value!
    //
    // Complete documentation can be found here
    // https://docs.opencv.org/4.6.0/d0/d2e/classcv_1_1CommandLineParser.html
    // ============================================================================

    const std::string keys =
        "{iterations iter  |100    | maximum number of iterations }"
        "{help h usage ?   |       | print this message   }"
        "{input i          |<none> | input image file     }"
        "{k                |<none> | number of clusters   }"
        "{type t           |0      | Type of kmeans used (0=OpenCV, 1=Perso, 2=Meanshift) }"
        "{hs               |20     | mean-shift spatial bandwidth (pixels) }"
        "{hc               |35     | mean-shift color bandwidth (0..255)   }"
        "{eps e            |0.5    | mean-shift convergence threshold      }"
        "{kmax             |20     | mean-shift max iterations per pixel   }"
        "{groundtruth gt   |       | ground truth segmentation image (optional) }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("K-means clustering application");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    const string imageFilename = parser.get<string>("input");
    const int k = parser.get<int>("k");
    const int type = parser.get<int>("type");
    const string groundTruthFilename = parser.get<string>("groundtruth");
    const int iterations = parser.get<int>("iterations");
    const float hs = parser.get<float>("hs");
    const float hc = parser.get<float>("hc");
    const float ms_eps = parser.get<float>("eps");
    const int kmax = parser.get<int>("kmax");

    if (!parser.check())
    {
        parser.printErrors();
        return EXIT_FAILURE;
    }

    // just for debugging, show the parsed arguments
    {
        cout << " Program called with the following arguments:" << endl;
        cout << " \timage file: " << imageFilename << endl;
        cout << " \tk: " << k << endl;
        if(!groundTruthFilename.empty()) cout << " \tground truth segmentation: " << groundTruthFilename << endl;
    }

    // load the color image to process from file
    Mat m;
    // for debugging use the macro PRINT_MAT_INFO to print the info about the matrix, like size and type
    PRINT_MAT_INFO(m);

    // 1) to call kmeans we need to first convert the image into floats (CV_32F)
    // see the method Mat.convertTo()
    m = imread(imageFilename);
    PRINT_MAT_INFO(m);
    Mat convertM;
    m.convertTo(convertM,CV_32F);

    // 2) kmeans asks for a mono-dimensional list of "points". Our "points" are the pixels of the image that can be seen as 3D points
    // where each coordinate is one of the color channels (e.g. R, G, B). But they are organized as a 2D table, we need
    // to re-arrange them into a single vector.
    // see the method Mat.reshape(), it is similar to matlab's reshape function.
    int N = static_cast<int>(convertM.total());
    Mat labels;
    int shape[2]={N, 1};
    Mat v = convertM.reshape(3, 2, shape);
    double epsilon = 1e-4;
    Mat centers;
    Mat Labels;
    Mat mask255;
    //String utiles pour enregistrer les resultats dans res
    path inPath(imageFilename);
    std::string stem = inPath.stem().string();
    path outDir = path("../res");
    string typeStr;
    if (type == 0){
        typeStr = "opencv";
    } else if (type == 1){
        typeStr = "kmeans_perso";
    } else if (type == 2){
        typeStr = "meanshift";
    }

    // now we can call kmeans(...)
    if (type == 0){
        cout << "Running OpenCV kmeans..." << endl;
        kmeans(v,k,labels,TermCriteria(TermCriteria::MAX_ITER,iterations,epsilon),3,KMEANS_RANDOM_CENTERS,centers);
        Labels = labels.reshape(1,m.rows).clone();
        Labels.convertTo(mask255, CV_8U, 255.0);
    } else if (type == 1){
        cout << "Running Perso kmeans..." << endl;
        Labels = kmeans_perso(m,k,iterations).clone();
        Labels.convertTo(mask255, CV_8U, 255.0);
    } else if (type == 2){
        cout << "Running MeanShift..." << endl;
        if (m.empty()) {
            cerr << "Error: input image is empty." << endl;
            return EXIT_FAILURE;
        }
        Mat segColor = meanshift(m, hs, hc, ms_eps, kmax);
        //DEBUG
        PRINT_MAT_INFO(segColor);
        imwrite("../res/meanshift_" + stem + "_color.png", segColor);
        Mat segGray;
        cvtColor(segColor, segGray, COLOR_BGR2GRAY);
        //mask255
        threshold(segGray, mask255, 0, 255, THRESH_BINARY | THRESH_OTSU);
    } else {
        cout << "Invalid type parameter. Use 0 for OpenCV, 1 for Perso, 2 for MeanShift." << endl;
        return EXIT_FAILURE;
    }

    path outPath = outDir / (stem +"_" + typeStr + "_mask.png");
    imwrite(outPath.string(), mask255);
    if (!groundTruthFilename.empty()){
        Mat gt;
        Mat gt16 = imread(groundTruthFilename,IMREAD_GRAYSCALE);
        gt16.convertTo(gt, CV_8U);

        Mat pred_pos = (mask255>0);
        Mat gt_pos = (gt==0);
        if (type == 0) gt_pos = (gt>0);
        if (type == 1) gt_pos = (gt>0);
        if (type == 2) gt_pos = (gt>0);


        Mat TPmat = pred_pos & gt_pos;
        Mat FPmat = pred_pos & (~gt_pos);
        Mat FNmat = (~pred_pos) & gt_pos;

        double TP=static_cast<double>(countNonZero(TPmat));
        double FP=static_cast<double>(countNonZero(FPmat));
        double FN=static_cast<double>(countNonZero(FNmat));

        double P = div(TP,TP+FP);
        double S = div(TP,TP+FN);
        double DSC = div(2.0*TP,2.0*TP+FP+FN);
        cout<< "P = "<< P <<endl;
        cout<< "S = "<< S <<endl;
        cout<< "DSC = "<< DSC <<endl;
    }

    return EXIT_SUCCESS;
}
