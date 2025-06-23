#include "utils.h"

// Hàm tính toán histogram cho một ảnh trong không gian màu BGR
Mat myCalcHistogram(const string& imagePath) {
    Mat image = imread(imagePath, IMREAD_COLOR);
    if (image.empty()) {
        cout << "Could not open image!" << imagePath << endl;
        return Mat();
    }

    // Tính histogram sử dụng hàm calcHist 
    // Khởi tạo histogram cho các kênh blue, green, red
    Mat b_hist, g_hist, r_hist;

    // Hyperparemeter cho hàm calcHist
    int channels[] = { 0, 1, 2 }; // B, G, R
    int histSize = 256; // 0..255
    float range[] = { 0, 256 }; // 0..255
    const float* histRange = range; // chuyển range về con trỏ hằng số gán cho histRange (cho hàm calcHist)

    calcHist(&image, 1, &channels[0], Mat(), b_hist, 1, &histSize, &histRange, true, false);
    calcHist(&image, 1, &channels[1], Mat(), g_hist, 1, &histSize, &histRange, true, false);
    calcHist(&image, 1, &channels[2], Mat(), r_hist, 1, &histSize, &histRange, true, false);

    // Chuẩn hóa histogram
    normalize(b_hist, b_hist, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, 1, NORM_MINMAX, -1, Mat());

    // Gộp cả 3 kênh màu vào cùng một histogram
    Mat hist;
    hconcat(vector<Mat>{b_hist, g_hist, r_hist}, hist);

    hist = hist.reshape(1, 1); // [768 x 1]
    hist = hist.t(); // [1 x 768]

    return hist;
}

// Hàm lấy descriptors sử dụng sift
Mat computeSIFTDescriptors(const Ptr<SIFT>& sift, const string& imagePath, const Size& size) {
    //Ptr<SIFT> sift = SIFT::create();
    Mat image = imread(imagePath, IMREAD_COLOR);
    if (image.empty()) {
        cout << "Unable to read file " + imagePath << endl;
        return Mat();
    }
    Mat resizedImage;
    resize(image, resizedImage, size);
    vector<KeyPoint> keypoints;
    Mat descriptors;
    sift->detectAndCompute(resizedImage, noArray(), keypoints, descriptors);
    return descriptors;
}

// Hàm lấy descriptors sử dụng orb
Mat computeORBDescriptors(const Ptr<ORB>& orb, const string& imagePath, const Size& size) {
    //Ptr<ORB> orb = ORB::create();
    Mat image = imread(imagePath, IMREAD_COLOR);
    if (image.empty()) {
        cout << "Unable to read file " + imagePath << endl;
        return Mat();
    }
    Mat resized;
    resize(image, resized, size);
    vector<KeyPoint> keypoints;
    Mat descriptors;
    orb->detectAndCompute(resized, noArray(), keypoints, descriptors);
    return descriptors;
}

// Hàm phân tách thành một vector chuỗi dựa trên dấu phân cách
vector<string> split(const string& s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}


// Hàm chuyển đổi ma trận histogram thành chuỗi (lưu ý với hàm myCalcHistogram cho ra [1 x 768] [cols x rows])
// Tương ứng với 1 cột và 768 hàng
string histogramToString(const Mat& hist) {
    stringstream ss;
    for (int j = 0; j < hist.rows; ++j) {
        //mat.at<float>(a, b) -> a là rows, b là cols
        ss << hist.at<float>(j, 0);
        if (j != hist.rows - 1) ss << ";";
    }
    return ss.str();
}

// Chuyển đổi vector<pair<string, double>> thành vector<Mat>
vector<Mat> myCvt_pair_to_Mat(const vector<pair<string, double>>& imagePaths) {
    vector<Mat> images;

    // Duyệt qua vector cặp
    for (const auto& entry : imagePaths) {
        string imagePath = entry.first;
        Mat image = imread(imagePath, IMREAD_COLOR); // Đọc ảnh màu

        // Kiểm tra xem ảnh có được đọc thành công hay không
        if (image.empty()) {
            cout << "Could not read the image: " << imagePath << endl;
            continue;
        }

        // Thêm ảnh vào vector mới
        images.push_back(image);
    }

    return images;
}

// hiển thị hình ảnh của một vector ảnh nhưng trong cùng 1 cửa sổ
void showImages(const vector<Mat>& images, const string& nameWindow) {
    int imagesPerRow = 7; // Số ảnh trên một hàng
    int rows = (int(images.size()) + imagesPerRow - 1) / imagesPerRow;

    // Xác định kích thước của ảnh con để tạo chiều dài nhỏ nhất và chiều rộng nhỏ nhất
    // dataset có kích thước [720 x 1280] -> [72 x 128] [144 x 256] [180 x 320] [216 x 384] [288 x 512]
    int minWidth = 144;
    int minHeight = 256;
    for (const auto& img : images)
    {
        if (img.rows < minHeight) minHeight = img.rows;
        if (img.cols < minWidth) minWidth = img.cols;
    }

    // Tạo figure theo kích thước
    int dstWidth = imagesPerRow * minWidth;
    int dstHeight = rows * minHeight;
    Mat dst = Mat(dstHeight, dstWidth, CV_8UC3, Scalar(0, 0, 0));

    // Copy các ảnh vào figure
    for (int i = 0; i < images.size(); ++i)
    {
        int x = (i % imagesPerRow) * minWidth;
        int y = (i / imagesPerRow) * minHeight;
        Rect roi(x, y, minWidth, minHeight);
        Mat resizedImg;
        resize(images[i], resizedImg, Size(minWidth, minHeight));
        resizedImg.copyTo(dst(roi));
    }

    // Hiển thị figure
    namedWindow(nameWindow, WINDOW_FULLSCREEN);
    imshow(nameWindow, dst);
    waitKey(0);
    // Đóng cửa sổ
    cv::destroyWindow(nameWindow);
}

// Hàm lấy tên file từ URL
string getFileNameFromFilePath(const string& filePath) {
    fs::path pathObj(filePath);
    string pictureName = pathObj.stem().string();
    return pictureName;
}

// Hàm đọc CSV và trả về một unordered_map với key là picture name và value là label
unordered_map<string, string> getNameLabelFromCSV(const string& csvFilePath) {
    unordered_map<string, string> pictureNameToLabel;
    ifstream file(csvFilePath);
    if (!file.is_open()) {
        cout << "Could not open the file " << csvFilePath << endl;
        return pictureNameToLabel;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string pictureName, label;

        // CSV có định dạng: picture name,label,...
        if (getline(ss, pictureName, ',') && getline(ss, label, ',')) {
            pictureNameToLabel[pictureName] = label;
        }
    }

    file.close();
    return pictureNameToLabel;
}

