#include "localFeature.h"

// Áp dụng kmeans vào để thực hiện truy vấn nhanh
// Hàm tạo từ điển
Mat SIFTFeature::createCodebook(int clusterCount) {
    Mat dictionary;
    Mat allDescriptors;
    for (const auto& r : records) {
        allDescriptors.push_back(r.descriptors);
    }
    allDescriptors.convertTo(allDescriptors, CV_32F);

    Mat labels, centers;
    // Trong Criteria, 50 là số lần lặp để sao cho cặp nhật lại các centers < 0.1 (độ chính xác)
    // Ở ngoài Criteria là 1 - số lần chạy tổng Kmeans và trả về centers, labels tốt nhất (ở đây train 50 lần kmeans x1)
    kmeans(allDescriptors, clusterCount, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 50, 0.1),
        1, KMEANS_RANDOM_CENTERS, centers);

    return centers;
}

// Gán đặc trưng vào cụm gần nhất
Mat SIFTFeature::mapFeaturesToCodewords(const Mat& descriptors, const Mat& dictionary) {
    Mat histogram;
    BFMatcher matcher(NORM_L2, false);
    //FlannBasedMatcher matcher;
    vector<DMatch> matches;
    histogram = Mat::zeros(1, dictionary.rows, CV_32F);

    // Thực hiện map đặc trưng trong từ điển tạo ra tần số của codeword
    // với mỗi match chuyển sang histogram (lấy match.trainIdx) +1
    matcher.match(descriptors, dictionary, matches);
    for (const auto& match : matches) {
        histogram.at<float>(0, match.trainIdx)++;
    }
    // Chuẩn hóa histogram
    normalize(histogram, histogram, 1, 0, NORM_L1);
    return histogram;
}

// Tạo biểu diễn histogram cho mỗi ảnh
vector<Mat> SIFTFeature::createHistograms(const Mat& dictionary) {
    vector<Mat> histograms;
    // Với mỗi descriptors có trong ảnh, sử dụng từ điển để tạo ra histogram
    for (const auto& r : records) {
        Mat histogram = this->mapFeaturesToCodewords(r.descriptors, dictionary);
        histograms.push_back(histogram);
    }
    return histograms;
}

// Truy vấn SIFT sử dụng Kmeans
vector<pair<string, double>> SIFTFeature::searchLocalKmeans(const string& queryImagePath, int n, const Size& size,
    const Mat& codebook, const vector<Mat>& histograms) {
    Ptr<SIFT> sift = SIFT::create();
    // 1. Thực hiện lấy words cho ảnh truy vấn
    Mat queryDescriptors = computeSIFTDescriptors(sift, queryImagePath, size);
    Mat queryHistogram = this->mapFeaturesToCodewords(queryDescriptors, codebook);

    // 2. Tính toán khoảng cách giữa words của ảnh truy vấn với các histogram trong bộ dữ liệu sử dụng norml2 (euclide)
    vector<pair<string, double>> imageSimilarities;
    int i = 0; //Dùng tạm biến đếm để lấy histogram trong vector
    for (const auto& r : this->records) {
        double distance = norm(queryHistogram, histograms[i++], NORM_L2);
        imageSimilarities.push_back(make_pair(r.URL, distance));
    }
    //cout << i << endl;

    // Sắp xếp theo khoảng cách tăng dần
    sort(imageSimilarities.begin(), imageSimilarities.end(), [](const pair<string, double>& a, const pair<string, double>& b) {
        return a.second < b.second;
        });

    // Chọn ra n ảnh
    if (imageSimilarities.size() > n) {
        imageSimilarities.resize(n);
    }

    return imageSimilarities;
}

// Ghi MAP cho 1 ảnh sử dụng SiftKmeans
void SIFTFeature::computeAndWriteMAP_Kmeans(const string& queryImagePath, const Size& size, ofstream& outputFile,
    const Mat& codebook, const vector<Mat>& histograms) {
    // 1. Khởi tạo MAPList để lưu vào và list returnImages
    vector<int> numberOfImagesList = { 3, 5, 11, 21 };
    vector<float> APs;

    // 2. Lấy label theo queryImage trong csvFile
    string queryLabel;
    bool foundQueryLabel = false;

    // 2.1 Tìm kiếm queryLabel trong records
    for (const auto& record : this->records) {
        if (record.URL == queryImagePath) {
            queryLabel = record.Label;
            foundQueryLabel = true;
            break;
        }
    }

    // 2.2 Không tìm thấy label trong file CSV
    if (!foundQueryLabel) {
        cout << "Query label not found in CSV file." << endl;
        return;
    }

    // 3. Thực hiện truy vấn 1 lần duy nhất với 21 tấm ảnh vì lúc nào cũng trả ra top 21 tấm
    vector<pair<string, double>> top21imagePaths = this->searchLocalKmeans(queryImagePath, 22, size, codebook, histograms);
    // Xóa ảnh đầu tiên (vì ảnh đầu tiên luôn là ảnh query)
    if (!top21imagePaths.empty() && top21imagePaths.front().first == queryImagePath) {
        top21imagePaths.erase(top21imagePaths.begin());
    }
    // Lấy top 3 ảnh
    vector<pair<string, double>> top3imagePaths(top21imagePaths.begin(), top21imagePaths.begin() + 3);
    // Lấy top 5 ảnh
    vector<pair<string, double>> top5imagePaths(top21imagePaths.begin(), top21imagePaths.begin() + 5);
    // Lấy top 11 ảnh
    vector<pair<string, double>> top11imagePaths(top21imagePaths.begin(), top21imagePaths.begin() + 11);

    // Làm thành một vector
    vector<vector<pair<string, double>>> top3_5_11_21 = { top3imagePaths, top5imagePaths, top11imagePaths, top21imagePaths };

    // 4. Với mỗi vector cặp đó thực hiện vòng lặp vector cặp lấy url ra và so sánh
    int i = 0; // biến đếm tạm
    for (const auto& vP : top3_5_11_21) {
        int n = numberOfImagesList[i++];
        int relevantCount = 0;
        float sumPrecision = 0.0;

        // For auto pair in vector pair
        for (int k = 0; k < vP.size(); ++k) {
            const auto& pair = vP[k];
            const string& url = pair.first;

            // Tìm đường dẫn URL tương ứng trong records
            auto it = std::find_if(this->records.begin(), this->records.end(), [&](const myRecord& r) {
                return r.URL == url;
                });

            if (it != this->records.end() && it->Label == queryLabel) {
                // Nếu label query giống với label trong record, thực hiện +relevantCount (groundtruth)
                // Giả sử tập top3 image = [A, B, A] và label = A
                // => 1/1 + 0 + 2/3 = 5/3
                // Tương tự với tập vector top5, 11...
                relevantCount++;
                sumPrecision += static_cast<float>(relevantCount) / (k + 1);
            }
        }

        // Tính AP và lưu vào vector APs
        // (5/3)/2 = 5/6 và lưu vào file
        float AP = (relevantCount == 0) ? 0.0f : sumPrecision / relevantCount;
        APs.push_back(AP);
    }

    string pictureName = getFileNameFromFilePath(queryImagePath);
    // 4. Ghi kết quả MAP vào file CSV (lưu ý append)
    outputFile << pictureName << "," << queryLabel << ",";
    for (float mapValue : APs) {
        outputFile << mapValue << ",";
    }
    outputFile << endl;
    return;
}

// Ghi MAP cho tất cả ảnh trong records sử dụng SiftKmeans
void SIFTFeature::processWriteMap_Kmeans(const string& MAPcsvPath, const Size& size, int maxRows,
    const Mat& codebook, const vector<Mat>& histograms) {
    // Mở file
    ofstream outputFile(MAPcsvPath, ios::app);
    if (!outputFile.good()) {
        cout << "Could not open file " << MAPcsvPath << endl;
        return;
    }
    outputFile << "Picture Name,Label,N=3,N=5,N=11,N=21\n"; //ghi header
    // Thực hiện ghi tiếp các dòng qua vòng lặp
    // Giới hạn số lần ghi vào file
    int count = 1;
    for (const auto& r : this->records) {
        auto begin = high_resolution_clock::now();
        this->computeAndWriteMAP_Kmeans(r.URL, size, outputFile, codebook, histograms);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);
        cout << count << " - Iter duration: " << duration.count() << "ms\n";
        if (count >= maxRows) {
            cout << "Rows Limit reached, stopping...\n";
            break;
        }
        count++;
    }
}

// ORB
// Hàm tạo từ điển
Mat ORBFeature::createCodebook(int clusterCount) {
    Mat dictionary;
    Mat allDescriptors;
    for (const auto& r : records) {
        allDescriptors.push_back(r.descriptors);
    }
    allDescriptors.convertTo(allDescriptors, CV_32F);

    Mat labels, centers;
    // Trong Criteria, 50 là số lần lặp để sao cho cặp nhật lại các centers < 0.1 (độ chính xác)
    // Ở ngoài Criteria là 1 - số lần chạy tổng Kmeans và trả về centers, labels tốt nhất (ở đây train 50 lần kmeans x1)
    kmeans(allDescriptors, clusterCount, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 50, 0.1),
        1, KMEANS_RANDOM_CENTERS, centers);

    return centers;
}

// Gán đặc trưng vào cụm gần nhất
Mat ORBFeature::mapFeaturesToCodewords(const Mat& descriptors, const Mat& dictionary) {
    Mat histogram;
    BFMatcher matcher(NORM_L2, false);
    vector<DMatch> matches;
    histogram = Mat::zeros(1, dictionary.rows, CV_32F);

    // Thực hiện map đặc trưng trong từ điển tạo ra tần số của codeword
    // với mỗi match chuyển sang histogram (lấy match.trainIdx) +1
    matcher.match(descriptors, dictionary, matches);
    for (const auto& match : matches) {
        histogram.at<float>(0, match.trainIdx)++;
    }
    // Chuẩn hóa histogram
    normalize(histogram, histogram, 1, 0, NORM_L1);
    return histogram;
}

// Tạo biểu diễn histogram cho mỗi ảnh
vector<Mat> ORBFeature::createHistograms(const Mat& dictionary) {
    vector<Mat> histograms;
    // Với mỗi descriptors có trong ảnh, sử dụng từ điển để tạo ra histogram
    for (const auto& r : records) {
        Mat histogram = this->mapFeaturesToCodewords(r.descriptors, dictionary);
        histograms.push_back(histogram);
    }
    return histograms;
}

// Truy vấn SIFT sử dụng Kmeans
vector<pair<string, double>> ORBFeature::searchLocalKmeans(const string& queryImagePath, int n, const Size& size,
    const Mat& codebook, const vector<Mat>& histograms) {
    Ptr<ORB> orb = ORB::create();
    // 1. Thực hiện lấy words cho ảnh truy vấn
    Mat queryDescriptors = computeORBDescriptors(orb, queryImagePath, size);
    Mat queryHistogram = this->mapFeaturesToCodewords(queryDescriptors, codebook);

    // 2. Tính toán khoảng cách giữa words của ảnh truy vấn với các histogram trong bộ dữ liệu sử dụng norml2 (euclide)
    vector<pair<string, double>> imageSimilarities;
    int i = 0; //Dùng tạm biến đếm để lấy histogram trong vector
    for (const auto& r : this->records) {
        double distance = norm(queryHistogram, histograms[i++], NORM_L2);
        imageSimilarities.push_back(make_pair(r.URL, distance));
    }
    //cout << i << endl;

    // Sắp xếp theo khoảng cách tăng dần
    sort(imageSimilarities.begin(), imageSimilarities.end(), [](const pair<string, double>& a, const pair<string, double>& b) {
        return a.second < b.second;
        });

    // Chọn ra n ảnh
    if (imageSimilarities.size() > n) {
        imageSimilarities.resize(n);
    }

    return imageSimilarities;
}

// Ghi MAP cho 1 ảnh sử dụng Orb
void ORBFeature::computeAndWriteMAP_Kmeans(const string& queryImagePath, const Size& size, ofstream& outputFile,
    const Mat& codebook, const vector<Mat>& histograms) {
    // 1. Khởi tạo MAPList để lưu vào và list returnImages
    vector<int> numberOfImagesList = { 3, 5, 11, 21 };
    vector<float> APs;

    // 2. Lấy label theo queryImage trong csvFile
    string queryLabel;
    bool foundQueryLabel = false;

    // 2.1 Tìm kiếm queryLabel trong records
    for (const auto& record : this->records) {
        if (record.URL == queryImagePath) {
            queryLabel = record.Label;
            foundQueryLabel = true;
            break;
        }
    }

    // 2.2 Không tìm thấy label trong file CSV
    if (!foundQueryLabel) {
        cout << "Query label not found in CSV file." << endl;
        return;
    }

    // 3. Thực hiện truy vấn 1 lần duy nhất với 21 tấm ảnh vì lúc nào cũng trả ra top 21 tấm
    vector<pair<string, double>> top21imagePaths = this->searchLocalKmeans(queryImagePath, 22, size, codebook, histograms);
    // Xóa ảnh đầu tiên (vì ảnh đầu tiên luôn là ảnh query)
    if (!top21imagePaths.empty() && top21imagePaths.front().first == queryImagePath) {
        top21imagePaths.erase(top21imagePaths.begin());
    }
    // Lấy top 3 ảnh
    vector<pair<string, double>> top3imagePaths(top21imagePaths.begin(), top21imagePaths.begin() + 3);
    // Lấy top 5 ảnh
    vector<pair<string, double>> top5imagePaths(top21imagePaths.begin(), top21imagePaths.begin() + 5);
    // Lấy top 11 ảnh
    vector<pair<string, double>> top11imagePaths(top21imagePaths.begin(), top21imagePaths.begin() + 11);

    // Làm thành một vector
    vector<vector<pair<string, double>>> top3_5_11_21 = { top3imagePaths, top5imagePaths, top11imagePaths, top21imagePaths };

    // 4. Với mỗi vector cặp đó thực hiện vòng lặp vector cặp lấy url ra và so sánh
    int i = 0; // biến đếm tạm
    for (const auto& vP : top3_5_11_21) {
        int n = numberOfImagesList[i++];
        int relevantCount = 0;
        float sumPrecision = 0.0;

        // For auto pair in vector pair
        for (int k = 0; k < vP.size(); ++k) {
            const auto& pair = vP[k];
            const string& url = pair.first;

            // Tìm đường dẫn URL tương ứng trong records
            auto it = std::find_if(this->records.begin(), this->records.end(), [&](const myRecord& r) {
                return r.URL == url;
                });

            if (it != this->records.end() && it->Label == queryLabel) {
                // Nếu label query giống với label trong record, thực hiện +relevantCount (groundtruth)
                // Giả sử tập top3 image = [A, B, A] và label = A
                // => 1/1 + 0 + 2/3 = 5/3
                // Tương tự với tập vector top5, 11...
                relevantCount++;
                sumPrecision += static_cast<float>(relevantCount) / (k + 1);
            }
        }

        // Tính AP và lưu vào vector APs
        // (5/3)/2 = 5/6 và lưu vào file
        float AP = (relevantCount == 0) ? 0.0f : sumPrecision / relevantCount;
        APs.push_back(AP);
    }

    string pictureName = getFileNameFromFilePath(queryImagePath);
    // 4. Ghi kết quả MAP vào file CSV (lưu ý append)
    outputFile << pictureName << "," << queryLabel << ",";
    for (float mapValue : APs) {
        outputFile << mapValue << ",";
    }
    outputFile << endl;
    return;
}

// Ghi MAP cho tất cả ảnh trong records sử dụng SiftKmeans
void ORBFeature::processWriteMap_Kmeans(const string& MAPcsvPath, const Size& size, int maxRows,
    const Mat& codebook, const vector<Mat>& histograms) {
    // Mở file
    ofstream outputFile(MAPcsvPath, ios::app);
    if (!outputFile.good()) {
        cout << "Could not open file " << MAPcsvPath << endl;
        return;
    }
    outputFile << "Picture Name,Label,N=3,N=5,N=11,N=21\n"; //ghi header
    // Thực hiện ghi tiếp các dòng qua vòng lặp
    // Giới hạn số lần ghi vào file
    int count = 1;
    for (const auto& r : this->records) {
        auto begin = high_resolution_clock::now();
        this->computeAndWriteMAP_Kmeans(r.URL, size, outputFile, codebook, histograms);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);
        cout << count << " - Iter duration: " << duration.count() << "ms\n";
        if (count >= maxRows) {
            cout << "Rows Limit reached, stopping...\n";
            break;
        }
        count++;
    }
}