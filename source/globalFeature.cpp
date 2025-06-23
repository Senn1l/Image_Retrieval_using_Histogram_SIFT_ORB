#include "globalFeature.h"

// Hàm để tính tổng của giá trị nhỏ nhất trong các bin tương ứng của hai histogram (độ đo)
double calcSimilarityScore_hist(const Mat& hist1, const Mat& hist2) {
    if (hist1.rows != hist2.rows || hist1.cols != hist2.cols) {
        cout << "Error: Histograms must have the same dimensions." << endl;
        return -1;
    }
    double similarityScore = 0.0;
    // [1 x 768] [cols x rows] -> so sánh rows
    for (int i = 0; i < hist1.rows; ++i) {
        similarityScore += min(hist1.at<float>(i), hist2.at<float>(i));
    }

    return similarityScore;
}

// Truy vấn sử dụng đặc trưng histogram và record
vector<pair<string, double>> HistogramFeature::searchGlobal(const string& queryImagePath, int n) {
    // Tính histogram của ảnh đầu vào
    Mat inputHist = myCalcHistogram(queryImagePath); // [1 x 768]

    vector<pair<string, double>> imageSimilarities; // Lưu trữ đường dẫn ảnh và độ tương đồng

    // Duyệt qua mỗi record lấy histogram và so sánh
    for (const auto& r : this->records) {
        double similarity = calcSimilarityScore_hist(inputHist, r.descriptors);
        imageSimilarities.push_back(make_pair(r.URL, similarity));
    }

    // Sắp xếp vector theo độ tương đồng giảm dần (càng lớn thì ảnh càng giống)
    sort(imageSimilarities.begin(), imageSimilarities.end(), [](const pair<string, double>& a, const pair<string, double>& b) {
        return a.second > b.second;
        });

    // Chọn ra n ảnh có độ tương đồng cao nhất
    if (imageSimilarities.size() > n) {
        imageSimilarities.resize(n);
    }

    //for (const auto& pair : imageSimilarities) {
    //    cout << "Image Path: " << pair.first << ", Similarity: " << pair.second << endl;
    //}

    return imageSimilarities;
}

// Tổng hợp n = 3, 5, 11, 21 ghi lại MAP vào một file CSV
// Có các cột là: URL,N=3,N=5,N=11,N=21
void HistogramFeature::computeAndWriteMAP(const string& queryImagePath, ofstream& outputFile) {
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

    // 3. Thực hiện truy vấn 1 lần duy nhất với 22 tấm ảnh vì lúc nào cũng trả ra top 22 tấm
    vector<pair<string, double>> top21imagePaths = this->searchGlobal(queryImagePath, 22);
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

    // 5. Ghi kết quả MAP vào file CSV (lưu ý append)
    outputFile << pictureName << "," << queryLabel << ",";
    for (float mapValue : APs) {
        outputFile << mapValue << ",";
    }
    outputFile << endl;

    return;
}

void HistogramFeature::processWriteMap(const string& MAPcsvPath, int maxRows) {
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
        this->computeAndWriteMAP(r.URL, outputFile);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);
        cout << count << " - Iter duration: " << duration.count() << "ms\n";
        if (count >= maxRows) {
            cout << "Rows Limit reached, stopping...\n";
            break;
        }
        count++;
    }
    outputFile.close();
}
